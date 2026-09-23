# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Fused virtual-query Spark Sol-Attn forward kernel for Hopper SM90.

Port of h3_sparse_attention/spark_reweight_sm120.py onto the WGMMA/TMA
SolAttnMainloopSm90 skeleton.  Native KC routing, the bitmask exact-block
stream (sol_attn/sm90/exact.py), and the Phase-1 route-mask decision logic
(SolAttnMainloopSm90.sol_attn_build_route_mask_from_acc) are reused as-is.
The approximate path runs a second QK WGMMA against query-parent conditioned
AK into a fresh accumulator, consumes parent-conditioned VC means (no block
length correction), and folds the per-parent virtual log mass into the route
column scratch.  Approximate and exact branches share one online softmax and
one output accumulator.

AK shares the single K smem stage but rides a dedicated one-stage TMA
pipeline with its own mbarriers: the shared K/V producer state advances once
per V load, so an extra K-phase per route group would desynchronize both
pipelines' parity discipline.  The K stage is reused for the first exact
block only after the AK WGMMA's warpgroup wait, which guarantees every warp
finished reading AK.  WGMMA accumulators are register-resident and the route
accumulator is dead once the mask is built, so the second QK accumulator
never overlaps it; peak register pressure matches the stock mainloop and
keeps sol_attn_mma_regs_override=128.
"""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace
from typing import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass import pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.utils import LayoutEnum

from sol_attn._vendor.flash_attn.cute import utils
from sol_attn._vendor.flash_attn.cute.block_info import BlockInfo
from sol_attn._vendor.flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from sol_attn._vendor.flash_attn.cute.mask import AttentionMask
from sol_attn._vendor.flash_attn.cute.seqlen_info import SeqlenInfoQK
from sol_attn._vendor.flash_attn.cute.softmax import Softmax
from sol_attn._vendor.flash_attn.cute import pipeline as pipeline_custom
from sol_attn.common import selector as sol_attn_selector
from sol_attn.sm90._compat import copy_utils, layout_utils, sm90_utils
from sol_attn.sm90 import exact as exact_stream
from sol_attn.sm90.mainloop import (
    SOL_ATTN_ROUTE_MASK_BARRIER_ID,
    SOL_ATTN_ROUTE_SUM_BARRIER_ID,
    SolAttnMainloopSm90,
)


class SparkReweightForwardSm90(SolAttnMainloopSm90):
    """Hopper fused virtual-query kernel for BF16 M64/N64/D128 inputs."""

    def __init__(
        self,
        tokens: int | None = None,
        *,
        external_route: bool = False,
        hybrid_route: bool = False,
        export_route: bool = False,
        force_local_blocks: bool = True,
        prefetch_approx_k: bool = True,
        shared_log_mass: bool = True,
        debug_route_trace: bool = False,
    ):
        if tokens is None:
            assume_lane_group_reduce = False
            assume_full_k_exact = False
            tail_exact_words1 = False
            assume_full_route_groups = False
            static_num_full_route_groups = -1
            static_tail_valid_count = -1
            tail_physical_tile16 = False
            exact_mask_seqlen_last_only = False
            tail16_lane_group_reduce = False
        else:
            blocks = (tokens + 63) // 64
            full_groups, tail = divmod(blocks, 64)
            has_full_groups = tail == 0
            has_full_blocks = tokens % 64 == 0
            assume_lane_group_reduce = has_full_blocks and has_full_groups
            assume_full_k_exact = has_full_blocks
            tail_exact_words1 = 0 < tail <= 8
            assume_full_route_groups = has_full_groups
            static_num_full_route_groups = -1 if has_full_groups else full_groups
            static_tail_valid_count = -1 if has_full_groups else tail
            tail_physical_tile16 = 0 < tail <= 16
            exact_mask_seqlen_last_only = not has_full_blocks
            tail16_lane_group_reduce = tail == 16
        super().__init__(
            cutlass.BFloat16,
            head_dim=128,
            head_dim_v=128,
            qhead_per_kvhead=1,
            is_causal=False,
            is_local=False,
            pack_gqa=False,
            tile_m=64,
            tile_n=64,
            num_stages=1,
            num_threads=128,
            sol_attn_assume_lane_group_route_reduce=assume_lane_group_reduce,
            sol_attn_assume_full_k_exact_blocks=assume_full_k_exact,
            sol_attn_tail_exact_words1=tail_exact_words1,
            sol_attn_assume_full_route_groups=assume_full_route_groups,
            sol_attn_static_num_full_route_groups=static_num_full_route_groups,
            sol_attn_static_tail_valid_count=static_tail_valid_count,
            sol_attn_tail_physical_tile16=tail_physical_tile16,
            sol_attn_exact_mask_seqlen_last_only=exact_mask_seqlen_last_only,
            sol_attn_tail16_lane_group_route_reduce=tail16_lane_group_reduce,
            sol_attn_num_splits=1,
            external_route=external_route,
            hybrid_route=hybrid_route,
            exact_only=False,
            export_route=export_route,
            force_local_blocks=force_local_blocks,
        )
        self.prefetch_approx_k = prefetch_approx_k
        self.shared_log_mass = shared_log_mass
        self.debug_route_trace = debug_route_trace

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct = [
            cute.struct.Align[
                cute.struct.MemRange[self.qk_dtype, cute.cosize(layout)], self.buffer_align_bytes
            ]
            for layout in (self.sQ_layout, self.sK_layout)
        ]
        sV_struct = cute.struct.Align[
            cute.struct.MemRange[self.pv_dtype, cute.cosize(self.sV_layout)],
            self.buffer_align_bytes,
        ]
        cosize_sQV = max(cute.cosize(self.sQ_layout), cute.cosize(self.sV_layout))
        sQV_struct = cute.struct.Align[cute.struct.MemRange[self.pv_dtype, cosize_sQV], 1024]
        cosize_sP = cute.cosize(self.sP_layout) if const_expr(self.sP_layout is not None) else 0
        sP_struct = cute.struct.Align[cute.struct.MemRange[self.pv_dtype, cosize_sP], 1024]
        route_mask_struct = cute.struct.Align[
            cute.struct.MemRange[Int32, 4], 16
        ]
        route_sums_struct = cute.struct.Align[
            cute.struct.MemRange[Float32, 4 * self.tile_n], 16
        ]
        mbar_ptr_Q_struct = cute.struct.MemRange[cutlass.Int64, 1 * 2]
        mbar_ptr_K_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        mbar_ptr_V_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        mbar_ptr_AK_struct = cute.struct.MemRange[cutlass.Int64, 1 * 2]

        @cute.struct
        class SharedStorageQKV:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            mbar_ptr_AK: mbar_ptr_AK_struct
            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct
            sP: sP_struct
            route_mask: route_mask_struct
            route_sums: route_sums_struct

        @cute.struct
        class SharedStorageSharedQV:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            mbar_ptr_AK: mbar_ptr_AK_struct
            sQ: sQV_struct
            sK: sK_struct
            sP: sP_struct
            route_mask: route_mask_struct
            route_sums: route_sums_struct

        return SharedStorageQKV if const_expr(not self.Q_in_regs) else SharedStorageSharedQV

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
        mQ, mK, mV, mO, mKC, mVC, mAK = [
            assume_tensor_aligned(t) for t in (q, k, v, o, kc, vc, ak)
        ]
        mQ, mK, mV, mO, mKC, mVC, mAK = [
            layout_utils.select(t, [1, 3, 2, 0])
            for t in (mQ, mK, mV, mO, mKC, mVC, mAK)
        ]
        mGlobalThresh = layout_utils.select(threshold, [1, 2, 0])
        mRouteMask = layout_utils.select(route_mask, [1, 2, 0, 3])
        if const_expr(self.debug_route_trace):
            mLSE = lse
        else:
            mLSE = layout_utils.select(lse, [1, 2, 0])

        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_qk.size
        self.num_threads_per_warp_group = 128
        self.num_wg_mma = self.num_mma_threads // self.num_threads_per_warp_group
        assert self.num_wg_mma == 1
        self.num_threads = self.num_threads_per_warp_group
        self.num_epilogue_threads = self.num_mma_threads
        self.num_mma_regs, self.num_producer_regs = 256, 56
        if const_expr(self.sol_attn_mma_regs_override is not None):
            self.num_mma_regs = self.sol_attn_mma_regs_override
        self.use_scheduler_barrier = False
        self.use_tma_Q = True
        self.use_tma_O = True
        self.rescale_O_before_gemm = False
        self._setup_attributes()
        self.sQ_layout, self.sK_layout, self.sV_layout, self.sO_layout = [
            sm90_utils.make_smem_layout(mX.element_type, LayoutEnum.ROW_MAJOR, shape, stage)
            for mX, shape, stage in [
                (mQ, (self.tile_m, self.tile_hdim), None),
                (mK, (self.tile_n, self.tile_hdim), self.num_stages),
                (mV, (self.tile_n, self.tile_hdimv), self.num_stages),
                # sO holds the BF16 PV epilogue tile; derive it from V.
                (mV, (self.tile_m, self.tile_hdimv), None),
            ]
        ]
        self.sP_layout = None
        SharedStorage = self._get_shared_storage_cls()

        gmem_tiled_copy_Q = cpasync.CopyBulkTensorTileG2SOp()
        gmem_tiled_copy_KV = cpasync.CopyBulkTensorTileG2SOp()
        gmem_tiled_copy_O = cpasync.CopyBulkTensorTileS2GOp()
        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
            for name, mX, layout in [
                ("Q", mQ, self.sQ_layout),
                ("K", mK, self.sK_layout),
                ("V", mV, self.sV_layout),
            ]
        }
        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_Q,
            mQ,
            self.sQ_layout,
            (self.tile_m, self.tile_hdim),
        )
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_KV,
            mK,
            cute.select(self.sK_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdim),
            1,
        )
        tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_KV,
            mV,
            cute.select(self.sV_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdimv),
            1,
        )
        tma_atom_KC, tma_tensor_KC = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_KV,
            mKC,
            cute.select(self.sK_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdim),
            1,
        )
        tma_atom_VC, tma_tensor_VC = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_KV,
            mVC,
            cute.select(self.sV_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdimv),
            1,
        )
        tma_atom_AK, tma_tensor_AK = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_KV,
            mAK,
            cute.select(self.sK_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdim),
            1,
        )
        tma_atom_O, tma_tensor_O = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_O,
            mO,
            self.sO_layout,
            (self.tile_m, self.tile_hdimv),
        )
        softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(
            softmax_scale, self.score_mod
        )
        # The host ABI carries two sink scalars; the reused route-mask builder
        # consumes the stock packed 16/16-bit range (empty range is vacuous).
        sink_range = Int32(sink_start_block) | (Int32(sink_end_block) << Int32(16))

        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            tma_tensor_V,
            tma_tensor_KC,
            tma_tensor_VC,
            tma_tensor_AK,
            tma_tensor_O,
            mGlobalThresh,
            mRouteMask,
            mLSE,
            lm,
            mapping,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_KC,
            tma_atom_VC,
            tma_atom_AK,
            tma_atom_O,
            softmax_scale_log2,
            softmax_scale,
            sink_range,
            q_block_start,
            parent_start,
            head_start,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sO_layout,
            tiled_mma_qk,
            tiled_mma_pv,
            SharedStorage,
        ).launch(
            grid=(q_block_count, head_count, mQ.shape[3]),
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mKC: cute.Tensor,
        mVC: cute.Tensor,
        mAK: cute.Tensor,
        mO: cute.Tensor,
        mGlobalThresh: cute.Tensor,
        mRouteMask: cute.Tensor,
        mLSE: cute.Tensor,
        mLM: cute.Tensor,
        mMap: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_KC: cute.CopyAtom,
        tma_atom_VC: cute.CopyAtom,
        tma_atom_AK: cute.CopyAtom,
        tma_atom_O: cute.CopyAtom,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        sink_range: Int32,
        q_block_start: Int32,
        parent_start: Int32,
        head_start: Int32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        SharedStorage: cutlass.Constexpr[Callable],
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            for tma_atom in (
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                tma_atom_KC,
                tma_atom_VC,
                tma_atom_AK,
                tma_atom_O,
            ):
                cpasync.prefetch_descriptor(tma_atom)

        storage = cutlass.utils.SmemAllocator().allocate(SharedStorage)

        ThreadCooperativeGroup = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
        tma_warp = ThreadCooperativeGroup(1)
        mma_warps = ThreadCooperativeGroup(self.num_mma_threads // cute.arch.WARP_SIZE)
        pipeline_q = pipeline_custom.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_Q.data_ptr(),
            num_stages=1,
            producer_group=tma_warp,
            consumer_group=mma_warps,
            tx_count=self.tma_copy_bytes["Q"],
            defer_sync=True,
        )
        pipeline_k = pipeline_custom.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_K.data_ptr(),
            num_stages=self.num_stages,
            producer_group=tma_warp,
            consumer_group=mma_warps,
            tx_count=self.tma_copy_bytes["K"],
            defer_sync=True,
        )
        pipeline_v = pipeline_custom.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_V.data_ptr(),
            num_stages=self.num_stages,
            producer_group=tma_warp,
            consumer_group=mma_warps,
            tx_count=self.tma_copy_bytes["V"],
            defer_sync=True,
        )
        pipeline_ak = pipeline_custom.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_AK.data_ptr(),
            num_stages=1,
            producer_group=tma_warp,
            consumer_group=mma_warps,
            tx_count=self.tma_copy_bytes["K"],
            defer_sync=True,
        )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sVt = layout_utils.transpose_view(sV)
        sO = storage.sQ.get_tensor(sO_layout.outer, swizzle=sO_layout.inner, dtype=self.dtype)
        route_mask = storage.route_mask.get_tensor(cute.make_layout((4,)))
        route_sums = storage.route_sums.get_tensor(cute.make_layout((4, self.tile_n)))

        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            self.is_local,
            False,
            None,
            None,
            qhead_per_kvhead_packgqa=1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=None,
            mCuSeqlensK=None,
            mSeqUsedQ=None,
            mSeqUsedK=None,
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.tile_m,
            self.tile_n,
            window_size_left=None,
            window_size_right=None,
            qhead_per_kvhead_packgqa=1,
        )

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        cute.arch.setmaxregister_increase(self.num_mma_regs)
        self.spark_route_mainloop(
            tiled_mma_qk,
            tiled_mma_pv,
            mQ,
            mK,
            mV,
            mKC,
            mVC,
            mAK,
            mO,
            mLSE,
            mLM,
            mMap,
            sQ,
            sK,
            sV,
            sVt,
            sO,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_KC,
            tma_atom_VC,
            tma_atom_AK,
            tma_atom_O,
            pipeline_q,
            pipeline_k,
            pipeline_v,
            pipeline_ak,
            SeqlenInfoCls,
            AttentionMaskCls,
            mGlobalThresh,
            mRouteMask,
            route_mask,
            route_sums,
            softmax_scale_log2,
            softmax_scale,
            sink_range,
            block_info,
            q_block_start,
            parent_start,
            head_start,
        )

    @cute.jit
    def spark_route_mainloop(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mKC: cute.Tensor,
        mVC: cute.Tensor,
        mAK: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        mLM: cute.Tensor,
        mMap: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sVt: cute.Tensor,
        sO: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_KC: cute.CopyAtom,
        tma_atom_VC: cute.CopyAtom,
        tma_atom_AK: cute.CopyAtom,
        tma_atom_O: cute.CopyAtom,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        pipeline_ak: pipeline.PipelineAsync,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        mGlobalThresh: cute.Tensor,
        mRouteMask: cute.Tensor,
        route_mask: cute.Tensor,
        route_sums: cute.Tensor,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        sink_range: Int32,
        block_info: BlockInfo,
        q_block_start: Int32,
        parent_start: Int32,
        head_start: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        q_block_idx, head_idx_local, batch_idx = cute.arch.block_idx()
        m_block = cute.arch.make_warp_uniform(q_block_idx + q_block_start)
        parent_idx = Int32(mMap[m_block]) - parent_start
        summary_head = cute.arch.make_warp_uniform(head_idx_local)
        head_idx = cute.arch.make_warp_uniform(head_idx_local + head_start)
        batch_idx = cute.arch.make_warp_uniform(batch_idx)
        summary_batch = batch_idx * mLM.shape[1] + parent_idx

        q_producer_phase = Int32(1)
        q_consumer_phase = Int32(0)
        kv_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_stages
        )
        kv_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_stages
        )
        ak_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, 1
        )
        ak_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, 1
        )

        seqlen = SeqlenInfoCls(batch_idx)
        mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
        mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, head_idx]
        mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[None, None, head_idx]
        mKC_cur = mKC[None, None, head_idx, batch_idx]
        mVC_cur = mVC[None, None, summary_head, summary_batch]
        mAK_cur = mAK[None, None, summary_head, summary_batch]

        gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0))
        gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (None, 0))
        gV = cute.local_tile(mV_cur, (self.tile_n, self.tile_hdimv), (None, 0))
        gKC = cute.local_tile(mKC_cur, (self.tile_n, self.tile_hdim), (None, 0))
        gVC = cute.local_tile(mVC_cur, (self.tile_n, self.tile_hdimv), (None, 0))
        gAK = cute.local_tile(mAK_cur, (self.tile_n, self.tile_hdim), (None, 0))

        load_Q, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_Q, 0, cute.make_layout(1), gQ, sQ, single_stage=True
        )
        tma_load_K_fn, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_K, 0, cute.make_layout(1), gK, sK
        )
        tma_load_K_fn = copy_utils.tma_producer_copy_fn(tma_load_K_fn, pipeline_k)
        tma_load_V_fn, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_V, 0, cute.make_layout(1), gV, sV
        )
        tma_load_V_fn = copy_utils.tma_producer_copy_fn(tma_load_V_fn, pipeline_v)
        tma_load_KC_fn, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_KC, 0, cute.make_layout(1), gKC, sK
        )
        tma_load_KC_fn = copy_utils.tma_producer_copy_fn(tma_load_KC_fn, pipeline_k)
        tma_load_VC_fn, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_VC, 0, cute.make_layout(1), gVC, sV
        )
        tma_load_VC_fn = copy_utils.tma_producer_copy_fn(tma_load_VC_fn, pipeline_v)
        tma_load_AK_fn, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_AK, 0, cute.make_layout(1), gAK, sK
        )
        tma_load_AK_fn = copy_utils.tma_producer_copy_fn(tma_load_AK_fn, pipeline_ak)

        if warp_idx == Int32(0):
            pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
            load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))
        pipeline_q.consumer_wait_w_index_phase(0, q_consumer_phase)

        warp_group_thread_layout = cute.make_layout(
            1, stride=self.num_threads_per_warp_group
        )
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        wg_mma_qk = tiled_mma_qk.get_slice(warp_group_thread_layout(Int32(0)))
        wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(Int32(0)))
        _, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(
            wg_mma_qk, (self.tile_m, self.tile_n, self.tile_hdim), sQ, sK
        )
        mma_qk_fn = partial(
            self.sol_attn_qk_gemm_zero_init,
            tiled_mma_qk,
            (self.tile_m, self.tile_n),
            tSrQ,
            tSrK,
        )
        acc_O, tOrP, tOrVt = sm90_utils.partition_fragment_ABC(
            wg_mma_pv, (self.tile_m, self.tile_hdimv, self.tile_n), None, sVt
        )
        mma_pv_fn = partial(sm90_utils.gemm_w_idx, tiled_mma_pv, acc_O, tOrP, tOrVt)
        acc_O.fill(0.0)
        cS_route = cute.make_identity_tensor((self.tile_m, self.tile_n))
        tScS_route_mn = layout_utils.reshape_acc_to_mn(thr_mma_qk.partition_C(cS_route))
        mask = AttentionMaskCls(seqlen)
        mask_fn = partial(
            mask.apply_mask,
            batch_idx=batch_idx,
            head_idx=head_idx,
            m_block=m_block,
            thr_mma=thr_mma_qk,
            mask_causal=self.is_causal,
            mask_local=self.is_local,
            aux_tensors=None,
            fastdiv_mods=None,
        )
        softmax = Softmax.create(
            softmax_scale_log2,
            num_rows=acc_O.shape[0][0] * acc_O.shape[1],
            softmax_scale=softmax_scale,
        )
        softmax.row_max.fill(-Float32.inf)
        softmax.row_sum.fill(0.0)
        exact_mma_one_n_block = partial(
            self.mma_one_n_block,
            mma_qk_fn=mma_qk_fn,
            pipeline_k=pipeline_k,
            pipeline_v=pipeline_v,
            acc_O=acc_O,
            tOrP=tOrP,
            smem_copy_params=SimpleNamespace(smem_thr_copy_P=None, tPsP=None),
            softmax=softmax,
            score_mod_fn=None,
            score_scale_fn=None,
            check_inf=not self.sol_attn_assume_nonempty_rows,
        )
        n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
        route_block_count = n_block_max - n_block_min
        if const_expr(self.sol_attn_static_num_full_route_groups >= 0):
            num_full_route_groups = Int32(self.sol_attn_static_num_full_route_groups)
            if const_expr(self.sol_attn_static_tail_valid_count > 0):
                tail_valid_count = Int32(self.sol_attn_static_tail_valid_count)
            else:
                tail_valid_count = Int32(0)
        elif const_expr(self.sol_attn_assume_full_route_groups):
            num_full_route_groups = cute.ceil_div(
                route_block_count, self.sol_attn_group_size
            )
            tail_valid_count = Int32(0)
        else:
            num_full_route_groups = route_block_count // Int32(self.sol_attn_group_size)
            tail_valid_count = (
                route_block_count
                - num_full_route_groups * Int32(self.sol_attn_group_size)
            )
        num_route_groups = num_full_route_groups
        if tail_valid_count > Int32(0):
            num_route_groups += Int32(1)
        O_should_accumulate = self.sol_attn_neutral_softmax_state
        for group_iter in cutlass.range(num_route_groups, unroll=1):
            group_start = n_block_min + group_iter * Int32(self.sol_attn_group_size)
            route_valid_count = Int32(self.sol_attn_group_size)
            if const_expr(not self.sol_attn_assume_full_route_groups):
                if group_iter == num_full_route_groups and tail_valid_count > Int32(0):
                    route_valid_count = tail_valid_count
            route_col_offset = group_start - (
                group_start // Int32(self.tile_n)
            ) * Int32(self.tile_n)
            route_n_block = group_start - route_col_offset
            route_tile = route_n_block // Int32(self.tile_n)
            has_next_route_group = group_iter + Int32(1) < num_route_groups
            next_route_tile = Int32(-1)
            if has_next_route_group:
                next_route_tile = (
                    group_start + Int32(self.sol_attn_group_size)
                ) // Int32(self.tile_n)

            if warp_idx == Int32(0):
                if group_iter == Int32(0):
                    pipeline_k.producer_acquire(kv_producer_state)
                    tma_load_KC_fn(
                        src_idx=route_tile,
                        producer_state=kv_producer_state,
                    )
                else:
                    previous_group_had_exact = (
                        (route_mask[0] != Int32(0))
                        or (route_mask[1] != Int32(0))
                        or (route_mask[2] != Int32(0))
                        or (route_mask[3] != Int32(0))
                    )
                    if not previous_group_had_exact:
                        pipeline_k.producer_acquire(kv_producer_state)
                        tma_load_KC_fn(
                            src_idx=route_tile,
                            producer_state=kv_producer_state,
                        )
                pipeline_v.producer_acquire(kv_producer_state)
                tma_load_VC_fn(
                    src_idx=route_tile,
                    producer_state=kv_producer_state,
                )
                kv_producer_state.advance()

            pipeline_k.consumer_wait(
                kv_consumer_state,
                pipeline_k.consumer_try_wait(kv_consumer_state),
            )
            acc_S = mma_qk_fn(B_idx=kv_consumer_state.index, wg_wait=-1)
            warpgroup.wait_group(0)
            pipeline_k.consumer_release(kv_consumer_state)
            if const_expr(self.prefetch_approx_k):
                # Speculative AK into the freed K stage while the CTA reduces
                # and compacts routes; drained below for all-exact groups.
                if warp_idx == Int32(0):
                    pipeline_ak.producer_acquire(ak_producer_state)
                    tma_load_AK_fn(
                        src_idx=route_tile,
                        producer_state=ak_producer_state,
                    )
                    ak_producer_state.advance()
            mask0, mask1, mask2, mask3 = self.sol_attn_build_route_mask_from_acc(
                acc_S,
                route_sums,
                tScS_route_mn,
                m_block,
                group_start,
                route_valid_count,
                route_col_offset,
                seqlen,
                batch_idx,
                head_idx,
                mGlobalThresh,
                softmax_scale_log2,
                sink_range,
                mRouteMask=mRouteMask,
                assume_full_route_group=False,
                route_mask_words_override=2,
            )
            if tidx == Int32(0):
                route_mask[0] = mask0
                route_mask[1] = mask1
                route_mask[2] = mask2
                route_mask[3] = mask3
            cute.arch.barrier(
                barrier_id=SOL_ATTN_ROUTE_MASK_BARRIER_ID,
                number_of_threads=self.num_mma_threads,
            )
            mask0 = route_mask[0]
            mask1 = route_mask[1]
            mask2 = route_mask[2]
            mask3 = route_mask[3]
            if const_expr(self.shared_log_mass):
                # One LM load/scale per block in warp 0, broadcast to all
                # query rows through the route column scratch.  Exact and
                # out-of-range columns stay masked with -inf.
                if warp_idx == Int32(0):
                    for word in cutlass.range_constexpr(2):
                        off = Int32(word * 32) + lane
                        bits = sol_attn_selector.sol_attn_mask_word_constexpr(
                            mask0, mask1, mask2, mask3, word
                        )
                        exact = (bits & (Int32(1) << lane)) != Int32(0)
                        valid = True
                        if const_expr(not self.sol_attn_assume_full_route_groups):
                            valid = off < route_valid_count
                        column = -Float32.inf
                        if valid and not exact:
                            column = (
                                Float32(
                                    mLM[
                                        batch_idx,
                                        parent_idx,
                                        summary_head,
                                        group_start + off,
                                    ]
                                )
                                * Float32(1.4426950408889634)
                                / softmax_scale_log2
                            )
                        route_sums[0, route_col_offset + off] = column
                cute.arch.barrier(
                    barrier_id=SOL_ATTN_ROUTE_SUM_BARRIER_ID,
                    number_of_threads=self.num_mma_threads,
                )
            if const_expr(self.debug_route_trace):
                if warp_idx == Int32(0) and lane == Int32(0):
                    mLSE[batch_idx, m_block, head_idx, group_iter, 0] = mask0
                    mLSE[batch_idx, m_block, head_idx, group_iter, 1] = mask1

            exact_mask0 = mask0
            exact_mask1 = mask1
            exact_mask2 = mask2
            exact_mask3 = mask3
            first_exact_n_block = group_start
            first_exact_exists = (
                (mask0 != Int32(0))
                or (mask1 != Int32(0))
                or (mask2 != Int32(0))
                or (mask3 != Int32(0))
            )
            if mask0 != Int32(0):
                first_lowbit = mask0 & (Int32(0) - mask0)
                first_exact_n_block += sol_attn_selector.sol_attn_bfind_b32(first_lowbit)
                exact_mask0 = mask0 & (mask0 - Int32(1))
            elif mask1 != Int32(0):
                first_lowbit = mask1 & (Int32(0) - mask1)
                first_exact_n_block += Int32(32) + (
                    sol_attn_selector.sol_attn_bfind_b32(first_lowbit)
                )
                exact_mask1 = mask1 & (mask1 - Int32(1))
            elif mask2 != Int32(0):
                first_lowbit = mask2 & (Int32(0) - mask2)
                first_exact_n_block += Int32(64) + (
                    sol_attn_selector.sol_attn_bfind_b32(first_lowbit)
                )
                exact_mask2 = mask2 & (mask2 - Int32(1))
            elif mask3 != Int32(0):
                first_lowbit = mask3 & (Int32(0) - mask3)
                first_exact_n_block += Int32(96) + (
                    sol_attn_selector.sol_attn_bfind_b32(first_lowbit)
                )
                exact_mask3 = mask3 & (mask3 - Int32(1))

            if const_expr(self.sol_attn_assume_full_route_groups):
                route_has_approx = (mask0 != Int32(-1)) or (mask1 != Int32(-1))
            else:
                valid0 = route_valid_count
                if valid0 > Int32(32):
                    valid0 = Int32(32)
                valid_bits0 = Int32(0)
                if valid0 > Int32(0):
                    valid_bits0 = Int32(-1)
                    if valid0 < Int32(32):
                        valid_bits0 = (Int32(1) << valid0) - Int32(1)
                valid1 = route_valid_count - Int32(32)
                if valid1 < Int32(0):
                    valid1 = Int32(0)
                if valid1 > Int32(32):
                    valid1 = Int32(32)
                valid_bits1 = Int32(0)
                if valid1 > Int32(0):
                    valid_bits1 = Int32(-1)
                    if valid1 < Int32(32):
                        valid_bits1 = (Int32(1) << valid1) - Int32(1)
                route_has_approx = (
                    ((mask0 & valid_bits0) != valid_bits0)
                    or ((mask1 & valid_bits1) != valid_bits1)
                )
            route_has_approx = route_has_approx and const_expr(not self.exact_only)

            pipeline_v.consumer_wait(
                kv_consumer_state,
                pipeline_v.consumer_try_wait(kv_consumer_state),
            )
            if route_has_approx:
                if const_expr(not self.prefetch_approx_k):
                    if warp_idx == Int32(0):
                        pipeline_ak.producer_acquire(ak_producer_state)
                        tma_load_AK_fn(
                            src_idx=route_tile,
                            producer_state=ak_producer_state,
                        )
                        ak_producer_state.advance()
                pipeline_ak.consumer_wait(
                    ak_consumer_state,
                    pipeline_ak.consumer_try_wait(ak_consumer_state),
                )
                acc_A = mma_qk_fn(B_idx=kv_consumer_state.index, wg_wait=-1)
                warpgroup.wait_group(0)
                pipeline_ak.consumer_release(ak_consumer_state)
                ak_consumer_state.advance()
                # Refill the freed K stage with the first exact block so the
                # transfer overlaps the approximate softmax and PV below.
                if first_exact_exists and warp_idx == Int32(0):
                    pipeline_k.producer_acquire(kv_producer_state)
                    tma_load_K_fn(
                        src_idx=first_exact_n_block,
                        producer_state=kv_producer_state,
                    )
                if const_expr(self.shared_log_mass):
                    _spark_add_shared_column_mask(acc_A, route_sums, tScS_route_mn)
                else:
                    _spark_mask_approx_log_mass(
                        acc_A,
                        tScS_route_mn,
                        mLM,
                        batch_idx,
                        parent_idx,
                        summary_head,
                        group_start,
                        route_valid_count,
                        mask0,
                        mask1,
                        mask2,
                        mask3,
                        softmax_scale_log2,
                    )
                if O_should_accumulate:
                    row_scale = softmax.online_softmax(
                        acc_A,
                        is_first=False,
                        check_inf=not self.sol_attn_assume_nonempty_rows,
                    )
                    softmax.rescale_O(acc_O, row_scale)
                else:
                    row_scale = softmax.online_softmax(
                        acc_A,
                        is_first=True,
                        check_inf=not self.sol_attn_assume_nonempty_rows,
                    )
                tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_A)
                utils.cvt_f16(tOrP_acc, tOrP)
                if O_should_accumulate:
                    sm90_utils.gemm_w_idx(
                        tiled_mma_pv,
                        acc_O,
                        tOrP,
                        tOrVt,
                        zero_init=False,
                        B_idx=kv_consumer_state.index,
                        wg_wait=-1,
                    )
                else:
                    sm90_utils.gemm_w_idx(
                        tiled_mma_pv,
                        acc_O,
                        tOrP,
                        tOrVt,
                        zero_init=True,
                        B_idx=kv_consumer_state.index,
                        wg_wait=-1,
                    )
                warpgroup.wait_group(0)
                O_should_accumulate = True
            else:
                if const_expr(self.prefetch_approx_k):
                    pipeline_ak.consumer_wait(
                        ak_consumer_state,
                        pipeline_ak.consumer_try_wait(ak_consumer_state),
                    )
                    pipeline_ak.consumer_release(ak_consumer_state)
                    ak_consumer_state.advance()
                if first_exact_exists and warp_idx == Int32(0):
                    pipeline_k.producer_acquire(kv_producer_state)
                    tma_load_K_fn(
                        src_idx=first_exact_n_block,
                        producer_state=kv_producer_state,
                    )
            pipeline_v.consumer_release(kv_consumer_state)
            kv_consumer_state.advance()

            last_n_block = Int32(-1)
            if const_expr(
                (not self.sol_attn_assume_full_k_exact_blocks)
                or self.sol_attn_exact_mask_seqlen_last_only
            ):
                last_n_block = (
                    (seqlen.seqlen_k + Int32(self.tile_n - 1)) // Int32(self.tile_n)
                ) - Int32(1)
            if O_should_accumulate:
                (
                    kv_producer_state,
                    kv_consumer_state,
                    O_should_accumulate,
                    _,
                ) = exact_stream.consume_exact_blocks(
                    exact_mask0,
                    exact_mask1,
                    exact_mask2,
                    exact_mask3,
                    group_start,
                    seqlen,
                    kv_producer_state,
                    kv_consumer_state,
                    tma_load_K_fn,
                    tma_load_V_fn,
                    pipeline_k,
                    pipeline_v,
                    warp_idx == Int32(0),
                    mma_pv_fn,
                    exact_mma_one_n_block,
                    mask_fn,
                    None,
                    O_should_accumulate,
                    self.warp_scheduler_barrier_sync,
                    self.warp_scheduler_barrier_arrive,
                    not self.sol_attn_assume_full_k_exact_blocks,
                    False,
                    self.sol_attn_group_words,
                    last_n_block,
                    self.sol_attn_exact_mask_seqlen_last_only,
                    first_exact_n_block,
                    first_exact_exists,
                    next_route_tile,
                    tma_load_KC_fn,
                )
            else:
                (
                    kv_producer_state,
                    kv_consumer_state,
                    O_should_accumulate,
                    _,
                ) = exact_stream.consume_exact_blocks(
                    exact_mask0,
                    exact_mask1,
                    exact_mask2,
                    exact_mask3,
                    group_start,
                    seqlen,
                    kv_producer_state,
                    kv_consumer_state,
                    tma_load_K_fn,
                    tma_load_V_fn,
                    pipeline_k,
                    pipeline_v,
                    warp_idx == Int32(0),
                    mma_pv_fn,
                    exact_mma_one_n_block,
                    mask_fn,
                    None,
                    O_should_accumulate,
                    self.warp_scheduler_barrier_sync,
                    self.warp_scheduler_barrier_arrive,
                    not self.sol_attn_assume_full_k_exact_blocks,
                    True,
                    self.sol_attn_group_words,
                    last_n_block,
                    self.sol_attn_exact_mask_seqlen_last_only,
                    first_exact_n_block,
                    first_exact_exists,
                    next_route_tile,
                    tma_load_KC_fn,
                )

        pipeline_q.consumer_release_w_index(0)
        final_scale = softmax.finalize(sink_val=None)
        softmax.rescale_O(acc_O, final_scale)
        if const_expr(self.debug_route_trace):
            self.epilogue_one_warpgroup_tma_o(
                acc_O,
                softmax.row_sum,
                mO,
                None,
                sO,
                seqlen,
                tma_atom_O,
                tiled_mma_pv,
                tidx,
                m_block,
                head_idx,
                batch_idx,
            )
        else:
            self.epilogue_one_warpgroup_tma_o(
                acc_O,
                softmax.row_sum,
                mO,
                mLSE,
                sO,
                seqlen,
                tma_atom_O,
                tiled_mma_pv,
                tidx,
                m_block,
                head_idx,
                batch_idx,
            )


@cute.jit
def _spark_add_shared_column_mask(
    acc: cute.Tensor,
    route_sums: cute.Tensor,
    tScS_mn: cute.Tensor,
):
    acc_mn = layout_utils.reshape_acc_to_mn(acc)
    for i in cutlass.range(cute.size(acc_mn), unroll_full=True):
        col = tScS_mn[i][1]
        acc_mn[i] = Float32(acc_mn[i]) + Float32(route_sums[0, col])


@cute.jit
def _spark_mask_approx_log_mass(
    acc: cute.Tensor,
    tScS_mn: cute.Tensor,
    mLM: cute.Tensor,
    batch_idx: Int32,
    parent_idx: Int32,
    summary_head: Int32,
    group_start: Int32,
    valid_count: Int32,
    mask0: Int32,
    mask1: Int32,
    mask2: Int32,
    mask3: Int32,
    scale_log2: Float32,
):
    # VC already holds parent-conditioned means, so no block-length factor;
    # LM is natural-log while the accumulator is in unscaled-dot units.
    acc_mn = layout_utils.reshape_acc_to_mn(acc)
    for i in cutlass.range(cute.size(acc_mn), unroll_full=True):
        col = tScS_mn[i][1]
        valid = col < valid_count
        exact = False
        if valid:
            exact = sol_attn_selector.sol_attn_test_exact_bit_limited_words(
                mask0, mask1, mask2, mask3, col, 2
            )
        if (not valid) or exact:
            acc_mn[i] = -Float32.inf
        else:
            acc_mn[i] = Float32(acc_mn[i]) + (
                Float32(mLM[batch_idx, parent_idx, summary_head, group_start + col])
                * Float32(1.4426950408889634)
                / scale_log2
            )


__all__ = ["SparkReweightForwardSm90"]
