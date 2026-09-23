# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Fused virtual-query Spark Sol-Attn forward kernel for Blackwell SM100.

The warp-specialized tcgen05/TMEM skeleton, route/exact phase structure, and
TMEM helpers come from sol_attn.sm100.mainloop; the Spark deltas are ported
from h3_sparse_attention.spark_reweight_sm120.  Native KC routing and the
N128 exact-pair engine are retained; approximate keys (AK) and log mass (LM)
are query-parent conditioned, the approximate value operand is the VC slot
holding parent-conditioned value means, and both branches share one online
softmax state and one output accumulator.

TMEM budget: the 256-column allocation is score 0..127 (pair P packed into
columns 64..127) and O 128..255, so no second score accumulator fits.  The
approximate QK against AK therefore reuses the pair-score TMEM region.  Unlike
the SM120 register-fragment reuse that corrupted CuTe lowering, TMEM reuse
here is a pair-score pipeline generation: the AK QK producer_acquire waits for
the owners' completion-backed release of the KC score load, and the P store
still aliases only drained score columns ahead of the next QK on the single
tcgen05 issuer.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import sol_attn._vendor.flash_attn.cute.pipeline as fa_pipeline
from cutlass import BFloat16, Float32, Int32, Uint8
from cutlass.cute.nvgpu import cpasync, tcgen05
from sol_attn._vendor.flash_attn.cute.cute_dsl_utils import assume_tensor_aligned

from sol_attn.common import layout_utils
from sol_attn.common.selector import (
    sol_attn_popc_b32,
    sol_attn_route_is_exact,
)
from sol_attn.sm100 import math as mma_utils
from sol_attn.sm100.mainloop import (
    D,
    DV,
    LN2,
    LOG2E,
    M,
    N_MEMBER,
    N_PACK_HALF,
    O_OFFSET,
    O_PACKED_COLUMN_STRIDE,
    O_PACKED_WORDS_PER_ROW_PER_THREAD,
    PACK_PV_GATHER_TILE,
    PACK_PV_INST,
    PACK_PV_QUARTER_INST,
    PACK_PV_TILE,
    PACK_QK_GATHER_TILE,
    PACK_QK_INST,
    PACK_QK_QUARTER_INST,
    PACK_QK_TILE,
    PACKET_WORDS,
    PAIR_P_OFFSET,
    PAIR_SCORE_OFFSET,
    PAIR_STAGES,
    ROUTE_HALVES_PER_GROUP,
    ROUTE_INDEX_CAPACITY,
    ROUTE_MASK_WORDS,
    ROUTE_TILE_SIZE,
    SEMANTIC_ROW_OFFSET,
    THREADS,
    TMEM_COLS,
    SharedStorage,
    _cvt_bf16x2_f32,
    _load_pack_k_half,
    _load_pack_v_half,
    _store_global_u32_inline,
    _store_pair_probability_chunked_tmemp,
)
from sol_attn.sm100.softmax import (
    _load_m64_n128_score as _load_pair_score,
    _online_update_one_half as _online_update_pair,
    _rescale_m64_partial_o as _rescale_pair_o,
)
from sol_attn.sm100.tmem import load_m64_o_fp32_256b


class SparkReweightForwardSm100:
    """tcgen05 Spark kernel: parent-conditioned approx K/V, shared softmax."""

    def __init__(
        self,
        *,
        external_route: bool = False,
        hybrid_route: bool = False,
        export_route: bool = False,
        force_local_blocks: bool = True,
        prefetch_approx_k: bool = True,
        shared_log_mass: bool = True,
        debug_route_trace: bool = False,
    ):
        self.external_route = external_route
        self.hybrid_route = hybrid_route
        self.export_route = export_route
        self.force_local_blocks = force_local_blocks
        self.prefetch_approx_k = prefetch_approx_k
        self.shared_log_mass = shared_log_mass
        self.debug_route_trace = debug_route_trace

    @cute.kernel
    def kernel(
        self,
        tiled_pack_qk: cute.TiledMma,
        tiled_pack_pv: cute.TiledMma,
        tma_atom_q: cute.CopyAtom,
        mQ_mkl: cute.Tensor,
        tma_atom_pack_k: cute.CopyAtom,
        mPackK_nkl: cute.Tensor,
        tma_atom_pack_v: cute.CopyAtom,
        mPackV_nkl: cute.Tensor,
        tma_atom_kc: cute.CopyAtom,
        mKC_nkl: cute.Tensor,
        tma_atom_vc: cute.CopyAtom,
        mVC_nkl: cute.Tensor,
        tma_atom_ak: cute.CopyAtom,
        mAK_nkl: cute.Tensor,
        mThreshold_bnh: cute.Tensor,
        mRouteMask: cute.Tensor,
        mO_bthd: cute.Tensor,
        mLSE: cute.Tensor,
        mLM: cute.Tensor,
        mMap: cute.Tensor,
        token_count: Int32,
        route_valid_total: Int32,
        num_route_tiles: Int32,
        softmax_scale: Float32,
        q_block_start: Int32,
        parent_start: Int32,
        head_start: Int32,
        sink_start_block: Int32,
        sink_end_block: Int32,
        q_layout: cute.ComposedLayout,
        pack_k_layout: cute.ComposedLayout,
        pack_k_gather_layout: cute.ComposedLayout,
        pack_p_layout: cute.ComposedLayout,
        pack_v_layout: cute.ComposedLayout,
        pack_v_gather_layout: cute.ComposedLayout,
        route_k_layout: cute.ComposedLayout,
        route_v_layout: cute.ComposedLayout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        q_block_idx_raw, head_idx_raw, batch_idx_raw = cute.arch.block_idx()
        q_tile_idx = cute.arch.make_warp_uniform(
            Int32(q_block_idx_raw) + q_block_start
        )
        parent_idx = Int32(mMap[q_tile_idx]) - parent_start
        summary_head = cute.arch.make_warp_uniform(Int32(head_idx_raw))
        head_idx = cute.arch.make_warp_uniform(Int32(head_idx_raw) + head_start)
        batch_idx = Int32(batch_idx_raw)
        softmax_scale_log2 = softmax_scale * Float32(LOG2E)

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sFinalStats = storage.final_stats.get_tensor(
            cute.make_layout((M, 2))
        )
        route_partial = storage.route_partial.get_tensor(
            cute.make_layout((4, ROUTE_TILE_SIZE))
        )
        column_masks = storage.column_masks.get_tensor(
            cute.make_layout((ROUTE_TILE_SIZE,))
        )
        route_packet = storage.route_packet.get_tensor(
            cute.make_layout((PACKET_WORDS,))
        )
        route_indices = storage.route_indices.get_tensor(
            cute.make_layout((ROUTE_INDEX_CAPACITY,))
        )
        sQ = smem.allocate_tensor(
            element_type=BFloat16,
            layout=q_layout.outer,
            byte_alignment=128,
            swizzle=q_layout.inner,
        )
        sPackK = smem.allocate_tensor(
            element_type=BFloat16,
            layout=pack_k_layout.outer,
            byte_alignment=128,
            swizzle=pack_k_layout.inner,
        )
        sPackV = smem.allocate_tensor(
            element_type=BFloat16,
            layout=pack_v_layout.outer,
            byte_alignment=128,
            swizzle=pack_v_layout.inner,
        )
        # One independent physical N128 K stage and one N128 V stage.  Every
        # runtime route/exact transaction stays in this completion domain.
        sPackKGather = cute.make_tensor(
            cute.recast_ptr(
                sPackK.iterator, pack_k_gather_layout.inner, BFloat16
            ),
            pack_k_gather_layout.outer,
        )
        sPackVGather = cute.make_tensor(
            cute.recast_ptr(
                sPackV.iterator, pack_v_gather_layout.inner, BFloat16
            ),
            pack_v_gather_layout.outer,
        )
        # KC/VC, AK, and exact K/V have disjoint lifetimes within each runtime
        # group.  They reuse the same independent N128 K and V allocations
        # without a cross-operand alias barrier.
        sKC = cute.make_tensor(
            cute.recast_ptr(sPackK.iterator, route_k_layout.inner, BFloat16),
            route_k_layout.outer,
        )
        sVC = cute.make_tensor(
            cute.recast_ptr(sPackV.iterator, route_v_layout.inner, BFloat16),
            route_v_layout.outer,
        )

        tmem_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
        score_loaded_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=4 * 32
        )
        final_stats_ready_barrier = pipeline.NamedBarrier(
            barrier_id=3, num_threads=4 * 32
        )
        pack_score_loaded_barrier = pipeline.NamedBarrier(
            barrier_id=4, num_threads=4 * 32
        )
        # Without speculative AK prefetch, warp 5 joins this rendezvous and
        # gates its on-demand AK load on the route packet.
        route_packet_ready_barrier = pipeline.NamedBarrier(
            barrier_id=5,
            num_threads=5 * 32 if self.prefetch_approx_k else THREADS,
        )
        exact_pair_p_ready_barrier = pipeline.NamedBarrier(
            barrier_id=6, num_threads=5 * 32
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_barrier,
        )
        tmem.allocate(TMEM_COLS)

        one_thread = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        pack_owner_threads = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, 4 * 32
        )
        q_bytes = cute.size_in_bytes(
            BFloat16, cute.select(q_layout, mode=[0, 1, 2])
        )
        route_k_bytes = cute.size_in_bytes(
            BFloat16, cute.select(route_k_layout, mode=[0, 1, 2])
        )
        route_v_bytes = cute.size_in_bytes(
            BFloat16, cute.select(route_v_layout, mode=[0, 1, 2])
        )
        pack_k_bytes = cute.size_in_bytes(
            BFloat16, cute.select(pack_k_layout, mode=[0, 1, 2])
        )
        pack_v_bytes = cute.size_in_bytes(
            BFloat16, cute.select(pack_v_layout, mode=[0, 1, 2])
        )
        assert route_k_bytes == pack_k_bytes
        assert route_v_bytes == pack_v_bytes
        q_pipe = fa_pipeline.PipelineTmaUmma.create(
            num_stages=1,
            producer_group=one_thread,
            consumer_group=one_thread,
            tx_count=q_bytes,
            barrier_storage=storage.q_mbar_ptr.data_ptr(),
        )
        pack_k_pipe = fa_pipeline.PipelineTmaUmma.create(
            num_stages=PAIR_STAGES,
            producer_group=one_thread,
            consumer_group=one_thread,
            tx_count=pack_k_bytes,
            barrier_storage=storage.pack_k_mbar_ptr.data_ptr(),
        )
        pack_v_pipe = fa_pipeline.PipelineTmaUmma.create(
            num_stages=PAIR_STAGES,
            producer_group=one_thread,
            consumer_group=one_thread,
            tx_count=pack_v_bytes,
            barrier_storage=storage.pack_v_mbar_ptr.data_ptr(),
        )
        pair_score_pipe = fa_pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=one_thread,
            consumer_group=pack_owner_threads,
            barrier_storage=storage.pair_score_mbar_ptr.data_ptr(),
        )
        pair_o_pipe = fa_pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=one_thread,
            consumer_group=pack_owner_threads,
            barrier_storage=storage.pair_o_mbar_ptr.data_ptr(),
        )

        mQ_cur = mQ_mkl[None, None, head_idx, batch_idx]
        mPackK_cur = mPackK_nkl[None, None, head_idx, batch_idx]
        mPackV_cur = mPackV_nkl[None, None, head_idx, batch_idx]
        mKC_cur = mKC_nkl[None, None, head_idx, batch_idx]
        summary_batch = batch_idx * mLM.shape[1] + parent_idx
        mVC_cur = mVC_nkl[None, None, summary_head, summary_batch]
        mAK_cur = mAK_nkl[None, None, summary_head, summary_batch]
        gQ = cute.local_tile(mQ_cur, (M, D), (None, 0))
        gPackK = cute.local_tile(
            mPackK_cur, (N_MEMBER, 64), (None, None)
        )
        gPackV = cute.local_tile(
            mPackV_cur, (64, N_MEMBER), (None, None)
        )
        gKC = cute.local_tile(mKC_cur, (N_PACK_HALF, D), (None, 0))
        gAK = cute.local_tile(mAK_cur, (N_PACK_HALF, D), (None, 0))
        gVC = cute.local_tile(mVC_cur, (DV, N_PACK_HALF), (0, None))
        thr_pack_qk = tiled_pack_qk.get_slice(0)
        thr_pack_pv = tiled_pack_pv.get_slice(0)
        tCgQ = thr_pack_qk.partition_A(gQ)
        tCgKC = thr_pack_qk.partition_B(gKC)
        tCgAK = thr_pack_qk.partition_B(gAK)
        tCgVC = thr_pack_pv.partition_B(gVC)
        tCrKC = tiled_pack_qk.make_fragment_B(sKC)
        tCrVC = tiled_pack_pv.make_fragment_B(sVC)
        tCrPackQ = tiled_pack_qk.make_fragment_A(sQ)
        tCrPackK = tiled_pack_qk.make_fragment_B(sPackK)
        tCrPackV = tiled_pack_pv.make_fragment_B(sPackV)

        tQsQ, tQgQ = cpasync.tma_partition(
            tma_atom_q,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tCgQ, 0, 3),
        )
        tPackKsK, tPackKgK = cpasync.tma_partition(
            tma_atom_pack_k,
            0,
            cute.make_layout(1),
            cute.group_modes(sPackKGather, 0, 3),
            cute.group_modes(gPackK, 0, 2),
        )
        tPackVsV, tPackVgV = cpasync.tma_partition(
            tma_atom_pack_v,
            0,
            cute.make_layout(1),
            cute.group_modes(sPackVGather, 0, 3),
            cute.group_modes(gPackV, 0, 2),
        )
        tKCsKC, tKCgKC = cpasync.tma_partition(
            tma_atom_kc,
            0,
            cute.make_layout(1),
            cute.group_modes(sKC, 0, 3),
            cute.group_modes(tCgKC, 0, 3),
        )
        tAKsKC, tAKgAK = cpasync.tma_partition(
            tma_atom_ak,
            0,
            cute.make_layout(1),
            cute.group_modes(sKC, 0, 3),
            cute.group_modes(tCgAK, 0, 3),
        )
        tVCsVC, tVCgVC = cpasync.tma_partition(
            tma_atom_vc,
            0,
            cute.make_layout(1),
            cute.group_modes(sVC, 0, 3),
            cute.group_modes(tCgVC, 0, 3),
        )

        pack_score_shape = tiled_pack_qk.partition_shape_C(
            PACK_QK_TILE[:2]
        )
        pack_score_template = tiled_pack_qk.make_fragment_C(pack_score_shape)
        pack_o_shape = tiled_pack_pv.partition_shape_C(PACK_PV_TILE[:2])
        pack_o_template = tiled_pack_pv.make_fragment_C(pack_o_shape)

        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(Float32)
        # The 256-column allocation leaves the second half of SM TMEM available
        # to another CTA.  The live allocation remains owned after release.
        tmem.relinquish_alloc_permit()
        tmem_base = tmem_ptr.toint()
        pair_tScore = cute.make_tensor(
            cute.make_ptr(
                Float32,
                tmem_base + Int32(PAIR_SCORE_OFFSET),
                cute.AddressSpace.tmem,
                assumed_align=16,
            ),
            pack_score_template.layout,
        )
        pair_tO = cute.make_tensor(
            cute.make_ptr(
                Float32,
                tmem_base + Int32(O_OFFSET),
                cute.AddressSpace.tmem,
                assumed_align=16,
            ),
            pack_o_template.layout,
        )
        # make_fragment_A drops the physical TMEM allocation base and addresses
        # packed BF16 columns in half-column units.  Restore both facts so
        # 2*tmem_base + 2*PAIR_P_OFFSET names columns 64..127.
        pair_tP_storage = cute.make_tensor(
            pair_tScore.iterator, pack_p_layout.outer
        )
        pair_tP_base = tiled_pack_pv.make_fragment_A(pair_tP_storage)[
            None, None, None, 0
        ]
        pair_tP = cute.make_tensor(
            pair_tP_base.iterator
            + tmem_base
            + tmem_base
            + Int32(PAIR_P_OFFSET * 2),
            pair_tP_base.layout,
        )
        q_producer = fa_pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, 1
        )
        q_consumer = fa_pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, 1
        )
        pack_k_producer = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, PAIR_STAGES
        )
        pack_k_consumer = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, PAIR_STAGES
        )
        pack_v_producer = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, PAIR_STAGES
        )
        pack_v_consumer = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, PAIR_STAGES
        )
        pair_score_producer = fa_pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, 1
        )
        pair_score_consumer = fa_pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, 1
        )
        pair_o_producer = fa_pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, 1
        )
        pair_o_consumer = fa_pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, 1
        )
        route_start_base = Int32(0)
        q_len = token_count - q_tile_idx * Int32(M)
        if q_len > Int32(M):
            q_len = Int32(M)
        threshold = Float32(
            mThreshold_bnh[batch_idx, q_tile_idx, head_idx]
        )

        if warp_idx == Int32(5):
            cpasync.prefetch_descriptor(tma_atom_q)
            cpasync.prefetch_descriptor(tma_atom_pack_k)
            cpasync.prefetch_descriptor(tma_atom_pack_v)
            cpasync.prefetch_descriptor(tma_atom_kc)
            cpasync.prefetch_descriptor(tma_atom_vc)
            cpasync.prefetch_descriptor(tma_atom_ak)

            q_pipe.producer_acquire(q_producer)
            q_barrier = q_pipe.producer_get_barrier(q_producer)
            cute.copy(
                tma_atom_q,
                tQgQ[(None, q_tile_idx)],
                tQsQ[(None, q_producer.index)],
                tma_bar_ptr=q_barrier,
            )
            q_producer.advance()

        is_owner = warp_idx >= Int32(1) and warp_idx <= Int32(4)
        is_score_consumer = warp_idx <= Int32(4)
        owner_tidx = tidx - Int32(32)

        # One register-resident online state and one TMEM-O initialization bit
        # span every route/exact transaction in every runtime group.
        running_max = -Float32.inf
        running_sum = Float32(0.0)
        owner_o_initialized = Int32(0)
        mma_o_initialized = Int32(0)

        if warp_idx == Int32(0):
            q_pipe.consumer_wait(q_consumer)

        # The outer loop owns one logical G256 exact-index lifetime.  The inner
        # loop consumes each physical score/PV half immediately; it appends only
        # integer indices, never a second score or probability fragment.
        num_logical_groups = (
            num_route_tiles + Int32(ROUTE_HALVES_PER_GROUP - 1)
        ) // Int32(ROUTE_HALVES_PER_GROUP)
        # BEGIN_G256_CURSOR_UNIFORM_INDUCTION
        # arch_make_warp_uniform is a lowering hint, not a value broadcast. Both
        # values are CTA-invariant integer scalars before the hint.
        logical_group_idx = cute.arch.make_warp_uniform(Int32(0))
        remaining_group_tiles = cute.arch.make_warp_uniform(num_route_tiles)
        while logical_group_idx < num_logical_groups:
            is_final_logical_group = (
                logical_group_idx + Int32(1) == num_logical_groups
            )
            group_route_tile_base = logical_group_idx * Int32(
                ROUTE_HALVES_PER_GROUP
            )
            physical_halves_this_group = remaining_group_tiles
            if physical_halves_this_group > Int32(ROUTE_HALVES_PER_GROUP):
                physical_halves_this_group = Int32(ROUTE_HALVES_PER_GROUP)

            for half_idx in cutlass.range(
                physical_halves_this_group, unroll=1
            ):
                route_tile_idx = cute.arch.make_warp_uniform(
                    group_route_tile_base + half_idx
                )
                is_final_route_tile = (
                    route_tile_idx + Int32(1) == num_route_tiles
                )
                is_logical_terminal_half = (
                    half_idx + Int32(1) == physical_halves_this_group
                )
                route_start = cute.arch.make_warp_uniform(
                    route_start_base
                    + route_tile_idx * Int32(ROUTE_TILE_SIZE)
                )
                remaining_route_count = cute.arch.make_warp_uniform(
                    route_valid_total
                    - route_tile_idx * Int32(ROUTE_TILE_SIZE)
                )
                valid_route_count = remaining_route_count
                if valid_route_count > Int32(ROUTE_TILE_SIZE):
                    valid_route_count = Int32(ROUTE_TILE_SIZE)
                if valid_route_count < Int32(0):
                    valid_route_count = Int32(0)

                # Native KC/VC fill the independent K/V stages.  The
                # parent-conditioned AK tile reuses the K stage after the KC
                # QK releases it; the speculative load is drained by warp 0 on
                # all-exact halves.
                if warp_idx == Int32(5):
                    pack_k_pipe.producer_acquire(pack_k_producer)
                    route_k_barrier = pack_k_pipe.producer_get_barrier(
                        pack_k_producer
                    )
                    cute.copy(
                        tma_atom_kc,
                        tKCgKC[(None, route_tile_idx)],
                        tKCsKC[(None, pack_k_producer.index)],
                        tma_bar_ptr=route_k_barrier,
                    )
                    pack_k_producer.advance()

                    pack_v_pipe.producer_acquire(pack_v_producer)
                    route_v_barrier = pack_v_pipe.producer_get_barrier(
                        pack_v_producer
                    )
                    cute.copy(
                        tma_atom_vc,
                        tVCgVC[(None, route_tile_idx)],
                        tVCsVC[(None, pack_v_producer.index)],
                        tma_bar_ptr=route_v_barrier,
                    )
                    pack_v_producer.advance()

                    if cutlass.const_expr(self.prefetch_approx_k):
                        pack_k_pipe.producer_acquire(pack_k_producer)
                        route_ak_barrier = pack_k_pipe.producer_get_barrier(
                            pack_k_producer
                        )
                        cute.copy(
                            tma_atom_ak,
                            tAKgAK[(None, route_tile_idx)],
                            tAKsKC[(None, pack_k_producer.index)],
                            tma_bar_ptr=route_ak_barrier,
                        )
                        pack_k_producer.advance()

                if warp_idx == Int32(0):
                    pack_k_pipe.consumer_wait(pack_k_consumer)
                    pair_score_pipe.producer_acquire(pair_score_producer)
                    mma_utils.gemm(
                        tiled_pack_qk,
                        pair_tScore,
                        tCrPackQ[None, None, None, q_consumer.index],
                        tCrKC[None, None, None, pack_k_consumer.index],
                        zero_init=True,
                    )
                    pair_score_pipe.producer_commit(pair_score_producer)
                    pair_score_producer.advance()
                    pack_k_pipe.consumer_release(pack_k_consumer)
                    pack_k_consumer.advance()

                # BEGIN_RUNTIME_GROUP_BODY

                # Route generation: the stock native-score reduction and
                # selector, except scores feed only the exact/approximate
                # decision; the approximate attention scores come from the AK
                # QK below.
                if is_owner:
                    pair_score_pipe.consumer_wait(pair_score_consumer)
                    score_raw, score_coords = _load_pair_score(
                        pack_score_template,
                        thr_pack_qk,
                        tmem_base,
                        Int32(PAIR_SCORE_OFFSET),
                        owner_tidx,
                    )
                    pack_score_loaded_barrier.arrive_and_wait()
                    pair_score_pipe.consumer_release(pair_score_consumer)
                    pair_score_consumer.advance()
                    owner_warp = owner_tidx // Int32(32)
                    lane = owner_tidx % Int32(32)
                    semantic_row = (
                        score_coords[0][0] + Int32(SEMANTIC_ROW_OFFSET)
                    ) & Int32(M - 1)
                    row_valid = semantic_row < q_len
                    lane_col_parity = (lane // Int32(2)) % Int32(2)
                    # Column-pair reduction: parity-0 lanes carry column 2*pair
                    # and parity-1 lanes carry column 2*pair+1.  The XOR-1/16/8/4
                    # butterfly tree never crosses lane column-parity classes,
                    # so one tree reduces both columns at once.  Writer lanes
                    # 0 and 2 equal 2*(col%2).  An external route mask makes
                    # the reduction dead: its sums only feed the selector.
                    if cutlass.const_expr(not self.external_route):
                        for pair_idx in cutlass.range_constexpr(
                            0, ROUTE_TILE_SIZE // 2, 2
                        ):
                            my_col0 = Int32(2 * pair_idx) + lane_col_parity
                            partial0 = Float32(0.0)
                            if row_valid and my_col0 < valid_route_count:
                                partial0 = Float32(score_raw[pair_idx])
                            my_col1 = Int32(2 * (pair_idx + 1)) + lane_col_parity
                            partial1 = Float32(0.0)
                            if row_valid and my_col1 < valid_route_count:
                                partial1 = Float32(score_raw[pair_idx + 1])

                            raw_partial0 = partial0
                            raw_partial1 = partial1
                            scaled0, scaled1 = cute.arch.mul_packed_f32x2(
                                (raw_partial0, raw_partial1),
                                (softmax_scale_log2, softmax_scale_log2),
                            )
                            peer_scaled0 = cute.arch.shuffle_sync_bfly(
                                scaled0, offset=1
                            )
                            peer_scaled1 = cute.arch.shuffle_sync_bfly(
                                scaled1, offset=1
                            )
                            partial0, partial1 = cute.arch.fma_packed_f32x2(
                                (raw_partial0, raw_partial1),
                                (softmax_scale_log2, softmax_scale_log2),
                                (peer_scaled0, peer_scaled1),
                            )
                            peer0 = cute.arch.shuffle_sync_bfly(
                                partial0, offset=16
                            )
                            peer1 = cute.arch.shuffle_sync_bfly(
                                partial1, offset=16
                            )
                            partial0, partial1 = cute.arch.add_packed_f32x2(
                                (partial0, partial1), (peer0, peer1)
                            )
                            peer0 = cute.arch.shuffle_sync_bfly(
                                partial0, offset=8
                            )
                            peer1 = cute.arch.shuffle_sync_bfly(
                                partial1, offset=8
                            )
                            partial0, partial1 = cute.arch.add_packed_f32x2(
                                (partial0, partial1), (peer0, peer1)
                            )
                            peer0 = cute.arch.shuffle_sync_bfly(
                                partial0, offset=4
                            )
                            peer1 = cute.arch.shuffle_sync_bfly(
                                partial1, offset=4
                            )
                            partial0, partial1 = cute.arch.add_packed_f32x2(
                                (partial0, partial1), (peer0, peer1)
                            )
                            if lane == Int32(0):
                                route_partial[owner_warp, 2 * pair_idx] = (
                                    partial0
                                )
                                route_partial[owner_warp, 2 * (pair_idx + 1)] = (
                                    partial1
                                )
                            if lane == Int32(2):
                                route_partial[owner_warp, 2 * pair_idx + 1] = (
                                    partial0
                                )
                                route_partial[
                                    owner_warp, 2 * (pair_idx + 1) + 1
                                ] = partial1

                    cute.arch.fence_view_async_shared()
                    score_loaded_barrier.arrive_and_wait()
                    if owner_warp == Int32(0):
                        mask0 = Int32(0)
                        mask1 = Int32(0)
                        mask2 = Int32(0)
                        mask3 = Int32(0)

                        # Half 0 starts a fresh G256 stream and half 1 appends
                        # to lane 0's cumulative packet word.
                        append_base = Int32(0)
                        if half_idx != Int32(0):
                            append_base = Int32(route_packet[6])

                        # A positive signed shift avoids materializing 1<<31:
                        # lane 0 gets zero and lane 31 gets 0x7fffffff.
                        lane_mask_lt = Int32(0x7FFFFFFF) >> (
                            Int32(31) - lane
                        )
                        preceding_word_count = Int32(0)
                        for word in cutlass.range_constexpr(ROUTE_MASK_WORDS):
                            off = Int32(word * 32) + lane
                            valid = off < valid_route_count
                            exact_pred = False
                            if valid:
                                if cutlass.const_expr(self.external_route):
                                    exact_pred = (
                                        Int32(
                                            mRouteMask[
                                                batch_idx,
                                                q_tile_idx,
                                                head_idx,
                                                route_start + off,
                                            ]
                                        )
                                        != Int32(0)
                                    ) and valid
                                else:
                                    pair_02 = Float32(
                                        route_partial[0, off]
                                    ) + Float32(route_partial[2, off])
                                    pair_13 = Float32(
                                        route_partial[1, off]
                                    ) + Float32(route_partial[3, off])
                                    col_mean = (pair_02 + pair_13) / Float32(
                                        q_len
                                    )
                                    if cutlass.const_expr(
                                        self.force_local_blocks
                                    ):
                                        exact_pred = sol_attn_route_is_exact(
                                            q_tile_idx,
                                            route_start + off,
                                            col_mean,
                                            threshold,
                                            valid,
                                        )
                                    else:
                                        exact_pred = (
                                            col_mean > threshold
                                        ) and valid
                                    # A NaN cutoff marks a numerically unsafe
                                    # Top-K boundary.  Hybrid mode uses the
                                    # explicit route for only that
                                    # query-block/head and leaves every other
                                    # row on the scalar-threshold selector.
                                    if cutlass.const_expr(self.hybrid_route):
                                        if threshold != threshold:
                                            exact_pred = (
                                                Int32(
                                                    mRouteMask[
                                                        batch_idx,
                                                        q_tile_idx,
                                                        head_idx,
                                                        route_start + off,
                                                    ]
                                                )
                                                != Int32(0)
                                            ) and valid
                                # Local retention is an attention-kernel
                                # policy, including hybrid rows whose selection
                                # came from an explicit mask.
                                if cutlass.const_expr(
                                    self.force_local_blocks
                                    and not self.external_route
                                ):
                                    distance = q_tile_idx - (route_start + off)
                                    exact_pred = exact_pred or (
                                        (distance >= -1) and (distance <= 1)
                                    )
                                # Sink is a KV-only contract.  Text queries
                                # remain a caller-side dense operation in
                                # MMDiT models.
                                exact_pred = (
                                    exact_pred
                                    or (
                                        route_start + off >= sink_start_block
                                        and route_start + off < sink_end_block
                                    )
                                )
                            if cutlass.const_expr(self.export_route):
                                if valid:
                                    mRouteMask[
                                        batch_idx,
                                        q_tile_idx,
                                        head_idx,
                                        route_start + off,
                                    ] = (
                                        Uint8(1) if exact_pred else Uint8(0)
                                    )
                            word_mask = Int32(
                                cute.arch.vote_ballot_sync(exact_pred)
                            )
                            if cutlass.const_expr(self.debug_route_trace):
                                if lane == Int32(0):
                                    mLSE[
                                        batch_idx,
                                        q_tile_idx,
                                        head_idx,
                                        route_tile_idx,
                                        word,
                                    ] = word_mask
                            if valid and not exact_pred:
                                if cutlass.const_expr(self.shared_log_mass):
                                    # One parent LM load per approximate block,
                                    # broadcast through the existing mask
                                    # scratch to every query row.  LM is in
                                    # natural-log units and the mask add lands
                                    # in unscaled-dot units.
                                    column_masks[off] = (
                                        Float32(
                                            mLM[
                                                batch_idx,
                                                parent_idx,
                                                summary_head,
                                                route_start + off,
                                            ]
                                        )
                                        * Float32(LOG2E)
                                        / softmax_scale_log2
                                    )
                                else:
                                    column_masks[off] = Float32(0.0)
                            else:
                                column_masks[off] = -Float32.inf
                            lane_rank = (
                                append_base
                                + preceding_word_count
                                + sol_attn_popc_b32(word_mask & lane_mask_lt)
                            )
                            if exact_pred:
                                route_indices[lane_rank] = route_start + off
                            if cutlass.const_expr(word == 0):
                                mask0 = word_mask
                            elif cutlass.const_expr(word == 1):
                                mask1 = word_mask
                            elif cutlass.const_expr(word == 2):
                                mask2 = word_mask
                            else:
                                mask3 = word_mask
                            preceding_word_count = (
                                preceding_word_count
                                + sol_attn_popc_b32(word_mask)
                            )

                        # Every selected lane has a unique rank; lane 0
                        # publishes the packet after reconvergence.
                        exact_count = preceding_word_count
                        if lane == Int32(0):
                            route_rank = append_base + exact_count

                            route_packet[0] = mask0
                            route_packet[1] = mask1
                            route_packet[2] = mask2
                            route_packet[3] = mask3
                            route_packet[4] = exact_count
                            route_packet[5] = append_base
                            route_packet[6] = route_rank
                            terminal_half_word = Int32(0)
                            if is_logical_terminal_half:
                                terminal_half_word = Int32(1)
                            route_packet[7] = terminal_half_word
                            cute.arch.fence_view_async_shared()

                    # The selector packet is now immutable until the next
                    # half's QK completion releases all owners.
                    score_loaded_barrier.arrive_and_wait()

                # Publish the route decision to warp 0 (and, without AK
                # prefetch, to the warp 5 producer).  The approximate P store
                # cannot precede this point: its scores do not exist until
                # warp 0 issues the AK QK.
                if cutlass.const_expr(self.prefetch_approx_k):
                    if is_score_consumer:
                        route_packet_ready_barrier.arrive_and_wait()
                else:
                    route_packet_ready_barrier.arrive_and_wait()

                if cutlass.const_expr(not self.prefetch_approx_k):
                    if warp_idx == Int32(5):
                        route_exact_count = Int32(route_packet[4])
                        if route_exact_count < valid_route_count:
                            pack_k_pipe.producer_acquire(pack_k_producer)
                            route_ak_barrier = pack_k_pipe.producer_get_barrier(
                                pack_k_producer
                            )
                            cute.copy(
                                tma_atom_ak,
                                tAKgAK[(None, route_tile_idx)],
                                tAKsKC[(None, pack_k_producer.index)],
                                tma_bar_ptr=route_ak_barrier,
                            )
                            pack_k_producer.advance()

                # Approximate phase 1: warp 0 re-runs QK against the
                # parent-conditioned keys into the same pair-score TMEM
                # region.  The producer_acquire is the completion-backed edge
                # that makes this accumulator reuse explicit.
                if warp_idx == Int32(0):
                    route_exact_count = Int32(route_packet[4])
                    route_has_approx = route_exact_count < valid_route_count
                    if route_has_approx:
                        pack_k_pipe.consumer_wait(pack_k_consumer)
                        pair_score_pipe.producer_acquire(pair_score_producer)
                        mma_utils.gemm(
                            tiled_pack_qk,
                            pair_tScore,
                            tCrPackQ[None, None, None, q_consumer.index],
                            tCrKC[None, None, None, pack_k_consumer.index],
                            zero_init=True,
                        )
                        pair_score_pipe.producer_commit(pair_score_producer)
                        pair_score_producer.advance()
                        pack_k_pipe.consumer_release(pack_k_consumer)
                        pack_k_consumer.advance()
                    elif cutlass.const_expr(self.prefetch_approx_k):
                        # All-exact half: drain the speculative AK stage so the
                        # exact-pair stream can reuse it.
                        pack_k_pipe.consumer_wait(pack_k_consumer)
                        pack_k_pipe.consumer_release(pack_k_consumer)
                        pack_k_consumer.advance()

                # Approximate phase 2: owners mask the AK scores (the shared
                # log mass already rides column_masks) and run the plain
                # online-softmax update.  VC holds parent-conditioned value
                # means, so no block-length multiplier is applied anywhere.
                if is_owner:
                    route_exact_count = Int32(route_packet[4])
                    has_route_approx = route_exact_count < valid_route_count
                    if has_route_approx:
                        pair_score_pipe.consumer_wait(pair_score_consumer)
                        approx_scores, approx_coords = _load_pair_score(
                            pack_score_template,
                            thr_pack_qk,
                            tmem_base,
                            Int32(PAIR_SCORE_OFFSET),
                            owner_tidx,
                        )
                        pack_score_loaded_barrier.arrive_and_wait()
                        pair_score_pipe.consumer_release(pair_score_consumer)
                        pair_score_consumer.advance()
                        semantic_row = (
                            approx_coords[0][0] + Int32(SEMANTIC_ROW_OFFSET)
                        ) & Int32(M - 1)
                        row_valid = semantic_row < q_len
                        row_mask = -Float32.inf
                        if row_valid:
                            row_mask = Float32(0.0)
                        assert cute.size(approx_scores) % 2 == 0
                        for i in cutlass.range_constexpr(
                            0, cute.size(approx_scores), 2
                        ):
                            column0 = approx_coords[i][1]
                            column1 = approx_coords[i + 1][1]
                            mask0 = Float32(column_masks[column0])
                            mask1 = Float32(column_masks[column1])
                            if cutlass.const_expr(not self.shared_log_mass):
                                block0 = route_start + column0
                                block1 = route_start + column1
                                lm0 = Float32(0.0)
                                lm1 = Float32(0.0)
                                if block0 < route_valid_total:
                                    lm0 = (
                                        Float32(
                                            mLM[
                                                batch_idx,
                                                parent_idx,
                                                summary_head,
                                                block0,
                                            ]
                                        )
                                        * Float32(LOG2E)
                                        / softmax_scale_log2
                                    )
                                if block1 < route_valid_total:
                                    lm1 = (
                                        Float32(
                                            mLM[
                                                batch_idx,
                                                parent_idx,
                                                summary_head,
                                                block1,
                                            ]
                                        )
                                        * Float32(LOG2E)
                                        / softmax_scale_log2
                                    )
                                mask0, mask1 = cute.arch.add_packed_f32x2(
                                    (mask0, mask1), (lm0, lm1)
                                )
                            mask0, mask1 = cute.arch.add_packed_f32x2(
                                (mask0, mask1), (row_mask, row_mask)
                            )
                            mask0, mask1 = cute.arch.add_packed_f32x2(
                                (
                                    Float32(approx_scores[i]),
                                    Float32(approx_scores[i + 1]),
                                ),
                                (mask0, mask1),
                            )
                            approx_scores[i] = mask0
                            approx_scores[i + 1] = mask1

                        probabilities, next_max, next_sum, row_alpha = (
                            _online_update_pair(
                                approx_scores,
                                running_max,
                                running_sum,
                                softmax_scale,
                            )
                        )
                        if owner_o_initialized != Int32(0):
                            _rescale_pair_o(
                                pack_o_template,
                                thr_pack_pv,
                                tmem_base,
                                Int32(O_OFFSET),
                                owner_tidx,
                                row_alpha,
                            )
                        _store_pair_probability_chunked_tmemp(
                            pack_o_template,
                            probabilities,
                            tmem_base,
                            Int32(PAIR_P_OFFSET),
                            owner_tidx,
                        )
                        # Publish P to warp 0 with the exact-path P-ready
                        # barrier; the approximate half is fully drained before
                        # the pre-exact CTA join, so the two uses never
                        # overlap.
                        exact_pair_p_ready_barrier.arrive_and_wait()
                        running_max = next_max
                        running_sum = next_sum
                        owner_o_initialized = Int32(1)

                # Approximate phase 3: one PV against the parent value means.
                # The route PV is deliberately drained before exact work so
                # all-exact, all-approx, odd, and partial-tail paths share one
                # phase boundary.
                if warp_idx == Int32(0):
                    route_exact_count = Int32(route_packet[4])
                    route_has_approx = route_exact_count < valid_route_count
                    pack_v_pipe.consumer_wait(pack_v_consumer)
                    if route_has_approx:
                        exact_pair_p_ready_barrier.arrive_and_wait()
                        mma_utils.gemm(
                            tiled_pack_pv,
                            pair_tO,
                            pair_tP,
                            tCrPackV[None, None, None, pack_v_consumer.index],
                            zero_init=mma_o_initialized == Int32(0),
                        )
                        # Half 0 is followed by half-1 route QK.  The terminal
                        # route half is followed by exact QK0 whenever the
                        # fused G256 index stream is nonempty.  Those score
                        # completions prove this PV complete; only a final
                        # route-only CTA needs an explicit O completion here.
                        if (
                            is_final_route_tile
                            and Int32(route_packet[6]) == Int32(0)
                        ):
                            pair_o_pipe.producer_commit(pair_o_producer)
                        mma_o_initialized = Int32(1)
                    pack_v_pipe.consumer_release(pack_v_consumer)
                    pack_v_consumer.advance()
                if is_owner:
                    cumulative_exact_count = Int32(route_packet[6])
                    if (
                        is_final_route_tile
                        and cumulative_exact_count == Int32(0)
                    ):
                        pair_o_pipe.consumer_wait(pair_o_consumer)

                # route_packet may be reused by the next physical half without
                # a CTA join.  Warp 0 reads this half's packet before it can
                # issue next-half QK; owner-warp 0 cannot overwrite the packet
                # until that QK's pair-score completion has released all
                # owners.

            # Both route halves have published their packet/index data and
            # drained approximate PV.  This is the only CTA-wide pre-exact join
            # in the logical G256 group; it publishes the combined list to
            # warp 5.
            cute.arch.barrier()
            # The cumulative count covers half 0 followed by half 1.  Pairing
            # this one ordered stream removes cross-half odd padding without
            # retaining either physical score fragment.
            exact_block_count = Int32(route_packet[6])
            exact_pair_count = (exact_block_count + Int32(1)) // Int32(2)
            pair_count = exact_pair_count
            has_pair_exact = exact_block_count > Int32(0)

            # BEGIN_GENERAL_N128_PAIR
            # Every executable exact count, including a logical-group terminal
            # exact1, stays in the N128 domain.

            # Warp 5 streams one physical N128 K stage and one physical N128 V
            # stage.  A missing odd peer duplicates block0 only for the
            # physical transaction; owners mask all upper-64 scores before
            # softmax.
            if warp_idx == Int32(5) and has_pair_exact:
                for pair_idx in cutlass.range(pair_count, unroll=1):
                    ordinal0 = pair_idx * Int32(2)
                    block0 = Int32(route_indices[ordinal0])
                    block1 = block0
                    if ordinal0 + Int32(1) < exact_block_count:
                        block1 = Int32(route_indices[ordinal0 + Int32(1)])

                    pack_k_pipe.producer_acquire(pack_k_producer)
                    pair_k_barrier = pack_k_pipe.producer_get_barrier(
                        pack_k_producer
                    )
                    _load_pack_k_half(
                        tma_atom_pack_k,
                        tPackKgK,
                        tPackKsK,
                        block0,
                        block1,
                        pack_k_producer.index * Int32(4),
                        pair_k_barrier,
                    )
                    pack_k_producer.advance()

                    pack_v_pipe.producer_acquire(pack_v_producer)
                    pair_v_barrier = pack_v_pipe.producer_get_barrier(
                        pack_v_producer
                    )
                    _load_pack_v_half(
                        tma_atom_pack_v,
                        tPackVgV,
                        tPackVsV,
                        block0,
                        block1,
                        pack_v_producer.index * Int32(4),
                        pair_v_barrier,
                    )
                    pack_v_producer.advance()

            if warp_idx == Int32(0) and has_pair_exact:
                # QK0 prologue.  K and score cursors advance exactly once per
                # QK; neither V nor O state is touched until the steady-state
                # PV path.
                pack_k_pipe.consumer_wait(pack_k_consumer)
                pair_score_pipe.producer_acquire(pair_score_producer)
                mma_utils.gemm(
                    tiled_pack_qk,
                    pair_tScore,
                    tCrPackQ[None, None, None, q_consumer.index],
                    tCrPackK[None, None, None, pack_k_consumer.index],
                    zero_init=True,
                )
                pair_score_pipe.producer_commit(pair_score_producer)
                pair_score_producer.advance()
                # PipelineTmaUmma release is tcgen05-completion-backed.
                pack_k_pipe.consumer_release(pack_k_consumer)
                pack_k_consumer.advance()

                for pair_idx in cutlass.range(pair_count, unroll=1):
                    # P aliases the drained upper half of S.  PV must therefore
                    # be issued before QK(i+1) overwrites S.  Both instructions
                    # are emitted back-to-back by warp 0, retaining the
                    # full-G128 tcgen05 dependency order without its K/V alias
                    # barriers.
                    pack_v_pipe.consumer_wait(pack_v_consumer)
                    # All four owners have completed their synchronous chunked
                    # TMEM stores and the helper's TMEM store fence before this
                    # five-warp rendezvous releases the single MMA warp.
                    exact_pair_p_ready_barrier.arrive_and_wait()
                    mma_utils.gemm(
                        tiled_pack_pv,
                        pair_tO,
                        pair_tP,
                        tCrPackV[None, None, None, pack_v_consumer.index],
                        zero_init=mma_o_initialized == Int32(0),
                    )
                    # QK(i+1) completion dominates PV(i) completion for every
                    # nonterminal transaction on this tcgen05 issuer.  Commit
                    # one explicit O-full generation only for the CTA's final
                    # PV.
                    if (
                        is_final_logical_group
                        and pair_idx + Int32(1) == pair_count
                    ):
                        pair_o_pipe.producer_commit(pair_o_producer)
                    mma_o_initialized = Int32(1)
                    pack_v_pipe.consumer_release(pack_v_consumer)
                    pack_v_consumer.advance()

                    if pair_idx + Int32(1) < pair_count:
                        pack_k_pipe.consumer_wait(pack_k_consumer)
                        pair_score_pipe.producer_acquire(pair_score_producer)
                        mma_utils.gemm(
                            tiled_pack_qk,
                            pair_tScore,
                            tCrPackQ[None, None, None, q_consumer.index],
                            tCrPackK[
                                None, None, None, pack_k_consumer.index
                            ],
                            zero_init=True,
                        )
                        pair_score_pipe.producer_commit(pair_score_producer)
                        pair_score_producer.advance()
                        pack_k_pipe.consumer_release(pack_k_consumer)
                        pack_k_consumer.advance()

            if is_owner and has_pair_exact:
                exact_owner_warp = owner_tidx // Int32(32)
                exact_lane = owner_tidx % Int32(32)
                for pair_idx in cutlass.range(pair_count, unroll=1):
                    ordinal0 = pair_idx * Int32(2)
                    block0 = Int32(route_indices[ordinal0])
                    has_peer = ordinal0 + Int32(1) < exact_block_count
                    block1 = block0
                    if has_peer:
                        block1 = Int32(route_indices[ordinal0 + Int32(1)])
                    valid0 = token_count - block0 * Int32(N_MEMBER)
                    valid1 = Int32(0)
                    if has_peer:
                        valid1 = token_count - block1 * Int32(N_MEMBER)
                    # Keep packed-select integer min/max lowering and
                    # exact-pair bookkeeping unchanged.
                    valid0 = max(Int32(0), min(valid0, Int32(N_MEMBER)))
                    valid1 = max(Int32(0), min(valid1, Int32(N_MEMBER)))

                    # Owner warp 0 builds two 64-column gates once for this
                    # exact N128 pair.  The score-load barrier below both
                    # protects the S/P alias and publishes these stores; no
                    # barrier or shared allocation is added.
                    if exact_owner_warp == Int32(0):
                        for cohort in cutlass.range_constexpr(4):
                            column = Int32(cohort * 32) + exact_lane
                            if cutlass.const_expr(cohort < 2):
                                if column >= valid0:
                                    column_masks[column] = -Float32.inf
                                else:
                                    column_masks[column] = Float32(0.0)
                            else:
                                if column - Int32(N_MEMBER) >= valid1:
                                    column_masks[column] = -Float32.inf
                                else:
                                    column_masks[column] = Float32(0.0)
                        cute.arch.fence_view_async_shared()

                    pair_score_pipe.consumer_wait(pair_score_consumer)
                    pair_scores, pair_coords = _load_pair_score(
                        pack_score_template,
                        thr_pack_qk,
                        tmem_base,
                        Int32(PAIR_SCORE_OFFSET),
                        owner_tidx,
                    )
                    # Every owner retires the complete score load before the
                    # packed P store aliases columns 64..127 of S.
                    pack_score_loaded_barrier.arrive_and_wait()
                    pair_score_pipe.consumer_release(pair_score_consumer)
                    pair_score_consumer.advance()

                    semantic_row = (
                        pair_coords[0][0] + Int32(SEMANTIC_ROW_OFFSET)
                    ) & Int32(M - 1)
                    row_valid = semantic_row < q_len
                    row_mask = -Float32.inf
                    if row_valid:
                        row_mask = Float32(0.0)
                    assert cute.size(pair_scores) % 2 == 0
                    for i in cutlass.range_constexpr(
                        0, cute.size(pair_scores), 2
                    ):
                        column0 = pair_coords[i][1]
                        column1 = pair_coords[i + 1][1]
                        mask0 = Float32(column_masks[column0])
                        mask1 = Float32(column_masks[column1])
                        mask0, mask1 = cute.arch.add_packed_f32x2(
                            (mask0, mask1), (row_mask, row_mask)
                        )
                        mask0, mask1 = cute.arch.add_packed_f32x2(
                            (
                                Float32(pair_scores[i]),
                                Float32(pair_scores[i + 1]),
                            ),
                            (mask0, mask1),
                        )
                        pair_scores[i] = mask0
                        pair_scores[i + 1] = mask1

                    probabilities, next_max, next_sum, row_alpha = (
                        _online_update_pair(
                            pair_scores,
                            running_max,
                            running_sum,
                            softmax_scale,
                        )
                    )
                    # For i>0, pair-score completion comes from QK(i), issued
                    # after PV(i-1) on the same tcgen05 issuer.  The score wait
                    # and load above therefore retire PV(i-1) before this O
                    # rescale.  Pair0 similarly follows either route QK or
                    # route PV->QK0.
                    if owner_o_initialized != Int32(0):
                        _rescale_pair_o(
                            pack_o_template,
                            thr_pack_pv,
                            tmem_base,
                            Int32(O_OFFSET),
                            owner_tidx,
                            row_alpha,
                        )
                    # The one TMEM P image is free once PV(i-1) completes.
                    # Keep probabilities FP32 until the live-range-bounded
                    # chunked R2T.
                    _store_pair_probability_chunked_tmemp(
                        pack_o_template,
                        probabilities,
                        tmem_base,
                        Int32(PAIR_P_OFFSET),
                        owner_tidx,
                    )
                    # The preceding helper performs tcgen05.wait::st for every
                    # chunk and a TMEM-store fence.  Publish P to warp 0 with
                    # one uniform generation shared by warps 0-4; warp 5 is
                    # excluded.
                    exact_pair_p_ready_barrier.arrive_and_wait()
                    running_max = next_max
                    running_sum = next_sum
                    owner_o_initialized = Int32(1)

                # There is no successor QK after the CTA's final exact PV.
                # Keep exactly one completion-backed wait before the epilogue;
                # all earlier groups flow into a successor route QK completion.
                if is_final_logical_group and pair_count > Int32(0):
                    pair_o_pipe.consumer_wait(pair_o_consumer)

            # route_indices reuse HB proof for the next logical group:
            # (1) warp 5 reads both indices before producing each pair's K/V,
            #     and final-pair score completion therefore dominates its last
            #     read;
            # (2) all owner index reads precede the final exact-P NamedBarrier;
            # (3) owner-warp0/lane0 is the sole next-group writer and reaches
            #     it only after that same exact loop.  For exact_count==0
            #     there are no readers.  Therefore no group-tail CTA barrier is
            #     required.

            # Cross-group progress is carried by the existing K/V buffer-free
            # phases and pair-score ready phase.  There is no CTA-wide
            # group-tail join: the next producer acquire cannot overwrite a
            # live K/V stage, and the next owner score load cannot precede QK
            # completion.
            # END_GENERAL_N128_PAIR

            logical_group_idx = cute.arch.make_warp_uniform(
                logical_group_idx + Int32(1)
            )
            remaining_group_tiles = cute.arch.make_warp_uniform(
                remaining_group_tiles - Int32(ROUTE_HALVES_PER_GROUP)
            )
            # END_RUNTIME_GROUP_BODY
        # END_G256_CURSOR_UNIFORM_INDUCTION

        if warp_idx == Int32(0):
            q_pipe.consumer_release(q_consumer)
            q_consumer.advance()

        if is_owner:
            lane = owner_tidx % Int32(32)
            owner_warp = owner_tidx // Int32(32)
            owner_row = (
                owner_warp * Int32(16)
                + lane // Int32(4)
                + (lane % Int32(2)) * Int32(8)
                + Int32(SEMANTIC_ROW_OFFSET)
            ) & Int32(M - 1)
            # Register state remains owner-local for the entire exact stream.
            # It is published only once here because the final Ld16x256b
            # epilogue remaps rows differently from the Ld16x64b xor-2 score
            # ownership.
            if (lane & Int32(2)) == Int32(0):
                sFinalStats[owner_row, 0] = running_sum
                sFinalStats[owner_row, 1] = running_max
            cute.arch.fence_view_async_shared()
            final_stats_ready_barrier.arrive_and_wait()

            o_regs, o_coords = load_m64_o_fp32_256b(
                pack_o_template,
                thr_pack_pv,
                tmem_base,
                owner_tidx,
            )
            assert cute.size(o_regs) == 64
            assert cute.size(o_coords) == 64

            # 4*w/4*w+1 belong to one semantic row and 4*w+2/4*w+3 to its
            # row-plus-eight peer.  Hoist validity, final-sum LDS, reciprocal,
            # and row base once per stratum.
            semantic_row0 = (
                owner_warp * Int32(16)
                + lane // Int32(4)
                + Int32(SEMANTIC_ROW_OFFSET)
            ) & Int32(M - 1)
            semantic_row1 = (semantic_row0 + Int32(8)) & Int32(M - 1)
            even_col_base = (lane % Int32(4)) * Int32(2)

            if semantic_row0 < q_len:
                inv_sum0 = cute.arch.rcp_approx(
                    Float32(sFinalStats[semantic_row0, 0])
                )
                query_idx0 = q_tile_idx * Int32(M) + semantic_row0
                destination_row0 = cute.domain_offset(
                    (batch_idx, query_idx0, head_idx, Int32(0)), mO_bthd
                )
                for word_i in cutlass.range(
                    O_PACKED_WORDS_PER_ROW_PER_THREAD, unroll_full=True
                ):
                    even_i = word_i * 4
                    odd_i = even_i + 1
                    even_value = Float32(o_regs[even_i]) * inv_sum0
                    odd_value = Float32(o_regs[odd_i]) * inv_sum0
                    packed_word = _cvt_bf16x2_f32(
                        Float32(odd_value), Float32(even_value)
                    )
                    even_col = (
                        even_col_base + word_i * O_PACKED_COLUMN_STRIDE
                    )
                    _store_global_u32_inline(
                        destination_row0.iterator + even_col, packed_word
                    )

            if semantic_row1 < q_len:
                inv_sum1 = cute.arch.rcp_approx(
                    Float32(sFinalStats[semantic_row1, 0])
                )
                query_idx1 = q_tile_idx * Int32(M) + semantic_row1
                destination_row1 = cute.domain_offset(
                    (batch_idx, query_idx1, head_idx, Int32(0)), mO_bthd
                )
                for word_i in cutlass.range(
                    O_PACKED_WORDS_PER_ROW_PER_THREAD, unroll_full=True
                ):
                    even_i = word_i * 4 + 2
                    odd_i = even_i + 1
                    even_value = Float32(o_regs[even_i]) * inv_sum1
                    odd_value = Float32(o_regs[odd_i]) * inv_sum1
                    packed_word = _cvt_bf16x2_f32(
                        Float32(odd_value), Float32(even_value)
                    )
                    even_col = (
                        even_col_base + word_i * O_PACKED_COLUMN_STRIDE
                    )
                    _store_global_u32_inline(
                        destination_row1.iterator + even_col, packed_word
                    )

            if cutlass.const_expr(not self.debug_route_trace):
                if (lane & Int32(2)) == Int32(0) and owner_row < q_len:
                    query_idx = q_tile_idx * Int32(M) + owner_row
                    mLSE[batch_idx, query_idx, head_idx] = (
                        running_max
                        + cute.math.log2(running_sum, fastmath=True)
                        * Float32(LN2)
                    )

        cute.arch.barrier()
        tmem.free(tmem_ptr)

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        o: cute.Tensor,
        kc: cute.Tensor,
        vc: cute.Tensor,
        threshold: cute.Tensor,
        route_mask: cute.Tensor,
        lse: cute.Tensor,
        ak: cute.Tensor,
        lm: cute.Tensor,
        mapping: cute.Tensor,
        q_block_start: Int32,
        q_block_count: Int32,
        parent_start: Int32,
        head_start: Int32,
        head_count: Int32,
        softmax_scale: Float32,
        sink_start_block: Int32,
        sink_end_block: Int32,
        stream: cuda.CUstream,
    ):
        q, k, v, o, kc, vc, ak = tuple(
            assume_tensor_aligned(t) for t in (q, k, v, o, kc, vc, ak)
        )
        q_mkl, k_nkl, kc_nkl = [
            layout_utils.select(t, [1, 3, 2, 0]) for t in (q, k, kc)
        ]
        v_nkl, vc_nkl = [
            layout_utils.select(t, [3, 1, 2, 0]) for t in (v, vc)
        ]
        ak_nkl = layout_utils.select(ak, [1, 3, 2, 0])
        token_count = cute.size(q_mkl.shape[0])
        num_blocks = cute.size(kc_nkl.shape[0])
        num_batches = cute.size(q_mkl.shape[3])
        num_route_tiles = cute.ceil_div(num_blocks, ROUTE_TILE_SIZE)
        pack_qk_op = tcgen05.MmaF16BF16Op(
            BFloat16,
            Float32,
            PACK_QK_INST,
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
        )
        tiled_pack_qk = cute.make_tiled_mma(pack_qk_op)
        pack_pv_op = tcgen05.MmaF16BF16Op(
            BFloat16,
            Float32,
            PACK_PV_INST,
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.TMEM,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.MN,
        )
        tiled_pack_pv = cute.make_tiled_mma(pack_pv_op)
        pack_qk_quarter_op = tcgen05.MmaF16BF16Op(
            BFloat16,
            Float32,
            PACK_QK_QUARTER_INST,
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
        )
        tiled_pack_qk_gather = cute.make_tiled_mma(pack_qk_quarter_op)
        pack_pv_quarter_op = tcgen05.MmaF16BF16Op(
            BFloat16,
            Float32,
            PACK_PV_QUARTER_INST,
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.TMEM,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.MN,
        )
        tiled_pack_pv_gather = cute.make_tiled_mma(pack_pv_quarter_op)
        q_layout = sm100_utils.make_smem_layout_a(
            tiled_pack_qk, PACK_QK_TILE, BFloat16, 1
        )
        pack_k_layout = sm100_utils.make_smem_layout_b(
            tiled_pack_qk, PACK_QK_TILE, BFloat16, PAIR_STAGES
        )
        pack_v_layout = sm100_utils.make_smem_layout_b(
            tiled_pack_pv, PACK_PV_TILE, BFloat16, PAIR_STAGES
        )
        pack_k_gather_layout = sm100_utils.make_smem_layout_b(
            tiled_pack_qk_gather,
            PACK_QK_GATHER_TILE,
            BFloat16,
            PAIR_STAGES * 4,
        )
        pack_v_gather_layout = sm100_utils.make_smem_layout_b(
            tiled_pack_pv_gather,
            PACK_PV_GATHER_TILE,
            BFloat16,
            PAIR_STAGES * 4,
        )
        pack_p_layout = sm100_utils.make_smem_layout_a(
            tiled_pack_pv, PACK_PV_TILE, BFloat16, 1
        )
        route_k_layout = sm100_utils.make_smem_layout_b(
            tiled_pack_qk, PACK_QK_TILE, BFloat16, PAIR_STAGES
        )
        route_v_layout = sm100_utils.make_smem_layout_b(
            tiled_pack_pv, PACK_PV_TILE, BFloat16, PAIR_STAGES
        )
        copy_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        q_tma_atom, q_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
            copy_op,
            q_mkl,
            cute.select(q_layout, mode=[0, 1, 2]),
            PACK_QK_TILE,
            tiled_pack_qk,
        )
        pack_k_tma_layout = cute.make_composed_layout(
            pack_k_gather_layout.inner,
            0,
            cute.make_layout((64, 64), stride=(64, 1)),
        )
        pack_k_tma_atom, pack_k_tma_tensor = cpasync.make_tiled_tma_atom(
            copy_op,
            k_nkl,
            pack_k_tma_layout,
            (64, 64),
        )
        pack_v_tma_layout = cute.make_composed_layout(
            pack_v_gather_layout.inner,
            0,
            cute.make_layout((64, 64), stride=(1, 64)),
        )
        pack_v_tma_atom, pack_v_tma_tensor = cpasync.make_tiled_tma_atom(
            copy_op,
            v_nkl,
            pack_v_tma_layout,
            (64, 64),
        )
        kc_tma_atom, kc_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
            copy_op,
            kc_nkl,
            cute.select(route_k_layout, mode=[0, 1, 2]),
            PACK_QK_TILE,
            tiled_pack_qk,
        )
        vc_tma_atom, vc_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
            copy_op,
            vc_nkl,
            cute.select(route_v_layout, mode=[0, 1, 2]),
            PACK_PV_TILE,
            tiled_pack_pv,
        )
        ak_tma_atom, ak_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
            copy_op,
            ak_nkl,
            cute.select(route_k_layout, mode=[0, 1, 2]),
            PACK_QK_TILE,
            tiled_pack_qk,
        )
        self.kernel(
            tiled_pack_qk,
            tiled_pack_pv,
            q_tma_atom,
            q_tma_tensor,
            pack_k_tma_atom,
            pack_k_tma_tensor,
            pack_v_tma_atom,
            pack_v_tma_tensor,
            kc_tma_atom,
            kc_tma_tensor,
            vc_tma_atom,
            vc_tma_tensor,
            ak_tma_atom,
            ak_tma_tensor,
            threshold,
            route_mask,
            o,
            lse,
            lm,
            mapping,
            Int32(token_count),
            Int32(num_blocks),
            Int32(num_route_tiles),
            softmax_scale,
            q_block_start,
            parent_start,
            head_start,
            sink_start_block,
            sink_end_block,
            q_layout,
            pack_k_layout,
            pack_k_gather_layout,
            pack_p_layout,
            pack_v_layout,
            pack_v_gather_layout,
            route_k_layout,
            route_v_layout,
        ).launch(
            grid=(q_block_count, head_count, num_batches),
            block=(THREADS, 1, 1),
            stream=stream,
            min_blocks_per_mp=2,
        )


__all__ = ["SparkReweightForwardSm100"]
