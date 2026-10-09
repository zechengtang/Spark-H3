# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
from types import SimpleNamespace
from typing import Type, Callable

import cuda.bindings.driver as cuda
import cutlass.cute.testing as testing
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import from_dlpack
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from sol_attn.common.selector import sol_attn_popc_b32

"""
A flash attention v2 forward pass example for NVIDIA Ampere SM80 architecture using CUTE DSL.

- Matrix Q is BxSqxNxH, B is batch dimension, Sq is query sequence length, N is number of heads, H is head dimension
- Matrix K is BxSkxNxH, B is batch dimension, Sk is key sequence length, N is number of heads, H is head dimension
- Matrix V is BxSkxNxH, B is batch dimension, Sk is key sequence length, N is number of heads, H is head dimension
- Matrix O is BxSqxNxH, B is batch dimension, Sq is query sequence length, N is number of heads, H is head dimension

This kernel supports the following features:
    - Utilizes CpAsync for efficient memory operations
    - Utilizes Ampere's tensor core for matrix multiply-accumulate (MMA) operations
    - Utilizes register pipeline to overlap shared memory-to-register transfers with computations.
    - Leverages DSL to implement an integrated online softmax fusion pattern.

This kernel works as follows:
1. Load Q and K matrices from global memory (GMEM) to shared memory (SMEM) using CpAsync operations.
2. Perform matrix multiply-accumulate (MMA) operations using tensor core instructions to compute intermediate result S.
3. Apply padding mask or causal mask to S during initial iterations.
4. Apply online softmax to S and rescale O using results from previous iteration.
5. Load V matrices and perform matrix multiply-accumulate (MMA) operations to compute final result O.
6. Normalize O after all iterations complete and store result back to global memory (GMEM).

To run this example:

.. code-block:: bash

    python examples/ampere/flash_attention_v2.py                                            \
      --dtype Float16 --head_dim 128 --m_block_size 128 --n_block_size 128                  \
      --num_threads 128 --batch_size 1 --seqlen_q 1280 --seqlen_k 1536                      \
      --num_head 16 --softmax_scale 1.0 --is_causal

The above command configures the model to use float16 for inputs and outputs. The problem dimensions
are set to: batch size of 1, query sequence length of 1280, key sequence length of 1536, head dimension
of 128, and 16 attention heads. The softmax scale is set to 1.0 and causal masking is enabled. The computation
uses tiles of size 128x128 for m and n dimensions, and utilizes 128 parallel threads.

To collect the performance with NCU profiler:

.. code-block:: bash

    ncu python examples/ampere/flash_attention_v2.py                                        \
        --dtype Float16 --head_dim 128 --m_block_size 128 --n_block_size 128                \
        --num_threads 128 --batch_size 1 --seqlen_q 1280 --seqlen_k 1536                    \
        --num_head 16 --softmax_scale 1.0 --is_causal --skip_ref_check

There are some constraints for this example:
* Only fp16 and bf16 data types are supported.
* The contiguous dimension of each tensor must be at least 16 bytes aligned.
* The log-sum-exp(for training) is not computed in the kernel.
* The values of `m_block_size`, `n_block_size`, and `head_dim` must be selected to stay within shared memory capacity limits.
* `m_block_size * 2` must be divisible by `num_threads`, otherwise the kernel will not be able to get the correct result.
"""


class FlashAttentionForwardAmpere:
    def __init__(
        self,
        head_dim: int,
        m_block_size: int = 128,
        n_block_size: int = 128,
        num_threads: int = 128,
        is_causal: bool = False,
    ):
        """Initializes the configuration for a flash attention v2 kernel.

        All contiguous dimensions must be at least 16 bytes aligned which indicates the head dimension
        should be a multiple of 8.

        :param head_dim: head dimension
        :type head_dim: int
        :param m_block_size: m block size
        :type m_block_size: int
        :param n_block_size: n block size
        :type n_block_size: int
        :param num_threads: number of threads
        :type num_threads: int
        :param is_causal: is causal
        """
        self._head_dim = head_dim
        self._m_block_size = m_block_size
        self._n_block_size = n_block_size
        # padding head_dim to a multiple of 32 as k_block_size
        self._head_dim_padded = (head_dim + 31) // 32 * 32
        self._num_threads = num_threads
        self._is_causal = is_causal

        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=num_threads
        )

    @staticmethod
    def can_implement(
        dtype, head_dim, m_block_size, n_block_size, num_threads, is_causal
    ) -> bool:
        """Check if the kernel can be implemented with the given parameters.

        :param dtype: data type
        :type dtype: cutlass.Numeric
        :param head_dim: head dimension
        :type head_dim: int
        :param m_block_size: m block size
        :type m_block_size: int
        :param n_block_size: n block size
        :type n_block_size: int
        :param num_threads: number of threads
        :type num_threads: int
        :param is_causal: is causal
        :type is_causal: bool

        :return: True if the kernel can be implemented, False otherwise
        :rtype: bool
        """
        # Check if data type is fp16 or bf16
        if dtype != cutlass.Float16 and dtype != cutlass.BFloat16:
            return False

        # Check if head dimension is a multiple of 8
        if head_dim % 8 != 0:
            return False

        # Check if number of threads is a multiple of 32
        if num_threads % 32 != 0:
            return False

        # Check if block size setting is out of shared memory capacity
        # Shared memory usage: Q tile + (K tile + V tile) where K and V use the same tile size
        smem_usage = (m_block_size * head_dim + n_block_size * head_dim * 2) * 2
        smem_capacity = utils.get_smem_capacity_in_bytes("sm_80")
        if smem_usage > smem_capacity:
            return False

        # Check if twice the block size is divisible by the number of threads
        if (m_block_size * 2) % num_threads != 0:
            return False

        return True

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        softmax_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        """Configures and launches the flash attention v2 kernel.

        mQ/mK/mV/mO has same data types(supports fp16 and bf16) and same layout:
        (batch_size, seqlen_q, num_head, head_dim):(seqlen_q * num_head * head_dim, num_head * head_dim, head_dim, 1)

        Prepares the shared memory layout, tiled copy atoms, tiled mma and shared memory storage.
        Then launches the kernel function with the prepared parameters.

        :param mQ: query tensor
        :type mQ: cute.Tensor
        :param mK: key tensor
        :type mK: cute.Tensor
        :param mV: value tensor
        :type mV: cute.Tensor
        :param mO: output tensor
        :type mO: cute.Tensor
        :param softmax_scale: softmax scale
        :type softmax_scale: cutlass.Float32
        """
        # Get the data type and check if it is fp16 or bf16
        if cutlass.const_expr(
            not (
                mQ.element_type == mK.element_type == mV.element_type == mO.element_type
            )
        ):
            raise TypeError("All tensors must have the same data type")
        if cutlass.const_expr(
            not (
                mQ.element_type == cutlass.Float16
                or mQ.element_type == cutlass.BFloat16
            )
        ):
            raise TypeError("Only Float16 or BFloat16 is supported")
        self._dtype: Type[cutlass.Numeric] = mQ.element_type
        # ///////////////////////////////////////////////////////////////////////////////
        # Shared memory layout: Q/K/V
        # ///////////////////////////////////////////////////////////////////////////////
        smem_k_block_size = 64 if self._head_dim_padded % 64 == 0 else 32
        swizzle_bits = 3 if smem_k_block_size == 64 else 2
        sQ_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size), stride=(smem_k_block_size, 1)),
        )
        sQ_layout = cute.tile_to_shape(
            sQ_layout_atom,
            (self._m_block_size, self._head_dim_padded),
            (0, 1),
        )

        sKV_layout_atom = sQ_layout_atom
        sKV_layout = cute.tile_to_shape(
            sKV_layout_atom,
            (self._n_block_size, self._head_dim_padded),
            (0, 1),
        )

        sO_layout = sQ_layout

        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sQ_layout)], 1024
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024
            ]

        # ///////////////////////////////////////////////////////////////////////////////
        # GMEM Tiled copy:
        # ///////////////////////////////////////////////////////////////////////////////
        # Thread layouts for copies
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self._dtype.width
        # atom_async_copy: async copy atom for QKV load
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        # atom_universal_copy: universal copy atom for O store
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        # tQKV_layout: thread layout for QKV load
        tQKV_shape_dim_1 = sQ_layout_atom.outer.shape[1] // async_copy_elems
        tQKV_layout = cute.make_layout(
            (self._num_threads // tQKV_shape_dim_1, tQKV_shape_dim_1),
            stride=(tQKV_shape_dim_1, 1),
        )
        # tO_layout: thread layout for O store
        tO_layout = tQKV_layout

        # Value layouts for copies
        vQKV_layout = cute.make_layout((1, async_copy_elems))
        vO_layout = vQKV_layout

        # gmem_tiled_copy_QKV: tiled copy for QKV load
        gmem_tiled_copy_QKV = cute.make_tiled_copy_tv(
            atom_async_copy, tQKV_layout, vQKV_layout
        )
        # gmem_tiled_copy_O: tiled copy for O store
        gmem_tiled_copy_O = cute.make_tiled_copy_tv(
            atom_universal_copy, tO_layout, vO_layout
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Tiled mma
        # ///////////////////////////////////////////////////////////////////////////////
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
            (self._num_threads // 32, 1, 1),
            permutation_mnk=(self._num_threads // 32 * 16, 16, 16),
        )

        # grid_dim: (m_block, batch_size, num_head)
        grid_dim = (
            cute.ceil_div(mQ.shape[1], self._m_block_size),
            cute.size(mQ.shape[0]),
            cute.size(mQ.shape[2]),
        )
        LOG2_E = 1.4426950408889634074
        softmax_scale_log2 = softmax_scale * LOG2_E
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            softmax_scale_log2,
            sQ_layout,
            sKV_layout,
            sO_layout,
            gmem_tiled_copy_QKV,
            gmem_tiled_copy_O,
            tiled_mma,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=[self._num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        sQ_layout: cute.ComposedLayout,
        sKV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_QKV: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        """Kernel function for flash attention v2.

        :param mQ: query tensor
        :type mQ: cute.Tensor
        :param mK: key tensor
        :type mK: cute.Tensor
        :param mV: value tensor
        :type mV: cute.Tensor
        :param mO: output tensor
        :type mO: cute.Tensor
        :param softmax_scale_log2: softmax scale log2
        :type softmax_scale_log2: cutlass.Float32
        :param sQ_layout: query layout
        :type sQ_layout: cute.ComposedLayout
        :param sKV_layout: key/value layout
        :type sKV_layout: cute.ComposedLayout
        :param sO_layout: output layout
        :type sO_layout: cute.ComposedLayout
        :param gmem_tiled_copy_QKV: tiled copy for QKV load
        :type gmem_tiled_copy_QKV: cute.TiledCopy
        :param gmem_tiled_copy_O: tiled copy for O store
        :type gmem_tiled_copy_O: cute.TiledCopy
        :param tiled_mma: tiled mma
        :type tiled_mma: cute.TiledMma
        :param SharedStorage: shared storage
        :type SharedStorage: cutlass.Constexpr
        """
        # Thread index, block index
        tidx, _, _ = cute.arch.thread_idx()
        m_block, batch_size, num_head = cute.arch.block_idx()

        n_block_max = cute.ceil_div(mK.shape[1], self._n_block_size)
        if self._is_causal:
            n_block_max = min(
                cute.ceil_div(
                    (m_block + 1) * self._m_block_size,
                    self._n_block_size,
                ),
                n_block_max,
            )
        n_block = n_block_max - 1

        # ///////////////////////////////////////////////////////////////////////////////
        # Get the appropriate tiles for this thread block.
        # ///////////////////////////////////////////////////////////////////////////////
        # (m_block_size, head_dim)
        gQ = cute.local_tile(
            mQ[batch_size, None, num_head, None],
            (self._m_block_size, self._head_dim_padded),
            (m_block, 0),
        )
        # (n_block_size, head_dim, n_block)
        gK = cute.local_tile(
            mK[batch_size, None, num_head, None],
            (self._n_block_size, self._head_dim_padded),
            (None, 0),
        )
        # (n_block_size, head_dim, n_block)
        gV = cute.local_tile(
            mV[batch_size, None, num_head, None],
            (self._n_block_size, self._head_dim_padded),
            (None, 0),
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Get shared memory buffer
        # ///////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()

        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sKV_layout)
        sV = storage.sV.get_tensor(sKV_layout)

        # Transpose view of V to tensor with layout (head_dim, n_block_size) for tiled mma
        sVt = cute.composition(
            sV,
            cute.make_layout(
                (self._head_dim_padded, self._n_block_size),
                stride=(self._n_block_size, 1),
            ),
        )

        gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_slice(tidx)
        # (CPY_Atom, CPY_M, CPY_K)
        tQgQ = gmem_thr_copy_QKV.partition_S(gQ)
        tQsQ = gmem_thr_copy_QKV.partition_D(sQ)
        # (CPY_Atom, CPY_N, CPY_K, n_block)
        tKgK = gmem_thr_copy_QKV.partition_S(gK)
        tKsK = gmem_thr_copy_QKV.partition_D(sK)
        # (CPY_Atom, CPY_N, CPY_K, n_block)
        tVgV = gmem_thr_copy_QKV.partition_S(gV)
        tVsV = gmem_thr_copy_QKV.partition_D(sV)

        # ///////////////////////////////////////////////////////////////////////////////
        # Tile MMA compute thread partitions and allocate accumulators
        # ///////////////////////////////////////////////////////////////////////////////
        thr_mma = tiled_mma.get_slice(tidx)
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
        tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))
        acc_shape_O = thr_mma.partition_shape_C(
            (self._m_block_size, self._head_dim_padded)
        )
        acc_O = cute.make_rmem_tensor(acc_shape_O, cutlass.Float32)
        acc_O.fill(0.0)

        # ///////////////////////////////////////////////////////////////////////////////
        # Smem copy atom tiling
        # ///////////////////////////////////////////////////////////////////////////////
        smem_copy_atom_Q = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self._dtype,
        )
        smem_copy_atom_K = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self._dtype,
        )
        smem_copy_atom_V = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            self._dtype,
        )
        smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom_Q, tiled_mma)
        smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom_K, tiled_mma)
        smem_tiled_copy_V = cute.make_tiled_copy_B(smem_copy_atom_V, tiled_mma)

        smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx)
        smem_thr_copy_K = smem_tiled_copy_K.get_slice(tidx)
        smem_thr_copy_V = smem_tiled_copy_V.get_slice(tidx)

        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
        tSsK = smem_thr_copy_K.partition_S(sK)
        tSrK_copy_view = smem_thr_copy_K.retile(tSrK)
        tOsVt = smem_thr_copy_V.partition_S(sVt)
        tOrVt_copy_view = smem_thr_copy_V.retile(tOrVt)

        # ///////////////////////////////////////////////////////////////////////////////
        # Predicate: Mark indices that need to copy when problem_shape isn't a multiple
        # of tile_shape
        # ///////////////////////////////////////////////////////////////////////////////
        # Construct identity layout for Q and KV
        mcQ = cute.make_identity_tensor(mQ.layout.shape)
        mcKV = cute.make_identity_tensor(mK.layout.shape)
        cQ = cute.local_tile(
            mcQ[batch_size, None, num_head, None],
            (self._m_block_size, self._head_dim_padded),
            (m_block, 0),
        )
        cKV = cute.local_tile(
            mcKV[batch_size, None, num_head, None],
            (self._n_block_size, self._head_dim_padded),
            (n_block, 0),
        )

        # Repeat the partitioning with identity layouts
        tQcQ = gmem_thr_copy_QKV.partition_S(cQ)
        tKVcKV = gmem_thr_copy_QKV.partition_S(cKV)
        # Allocate predicate tensors for m and n, here we only allocate the tile of k, and do special process for mn.
        # This is to reduce register pressure and gets 2-3% performance gain compared with allocating the whole tile.
        tQpQ = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tQsQ.shape[0][1],
                    cute.size(tQsQ, mode=[1]),
                    cute.size(tQsQ, mode=[2]),
                ),
                stride=(cute.size(tQsQ, mode=[2]), 0, 1),
            ),
            cutlass.Boolean,
        )
        tKVpKV = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tKsK.shape[0][1],
                    cute.size(tKsK, mode=[1]),
                    cute.size(tKsK, mode=[2]),
                ),
                stride=(cute.size(tKsK, mode=[2]), 0, 1),
            ),
            cutlass.Boolean,
        )
        # Set predicates for head_dim bounds, seqlen_q/k bounds is processed at the first tile.
        for rest_v in cutlass.range_constexpr(tQpQ.shape[0]):
            for rest_k in cutlass.range_constexpr(tQpQ.shape[2]):
                tQpQ[rest_v, 0, rest_k] = cute.elem_less(
                    tQcQ[(0, rest_v), 0, rest_k][3], mQ.layout.shape[3]
                )
        for rest_v in cutlass.range_constexpr(tKVpKV.shape[0]):
            for rest_k in cutlass.range_constexpr(tKVpKV.shape[2]):
                tKVpKV[rest_v, 0, rest_k] = cute.elem_less(
                    tKVcKV[(0, rest_v), 0, rest_k][3], mK.layout.shape[3]
                )
        # ///////////////////////////////////////////////////////////////////////////////
        # Prefetch Prologue
        # ///////////////////////////////////////////////////////////////////////////////
        # Start async loads of the last mn-tile, where we take care of the mn residue
        for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            if cute.elem_less(tQcQ[0, m, 0][1], mQ.layout.shape[1]):
                cute.copy(
                    gmem_tiled_copy_QKV,
                    tQgQ[None, m, None],
                    tQsQ[None, m, None],
                    pred=tQpQ[None, m, None],
                )
            else:
                # Clear the smem tiles to account for predicated off loads
                tQsQ[None, m, None].fill(0)
        for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
            if cute.elem_less(tKVcKV[0, n, 0][1], mK.layout.shape[1]):
                cute.copy(
                    gmem_tiled_copy_QKV,
                    tKgK[None, n, None, n_block],
                    tKsK[None, n, None],
                    pred=tKVpKV[None, n, None],
                )
            else:
                # Clear the smem tiles to account for predicated off loads
                tKsK[None, n, None].fill(0)

        cute.arch.cp_async_commit_group()
        # ///////////////////////////////////////////////////////////////////////////////
        # Softmax intermediate result: row_max and row_sum
        # ///////////////////////////////////////////////////////////////////////////////
        # shape: (atom_v_m * rest_m)
        row_max = cute.make_rmem_tensor(
            (acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32
        )
        # shape: (atom_v_m * rest_m)
        row_sum = cute.make_rmem_tensor(
            (acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32
        )
        row_max.fill(-cutlass.Float32.inf)
        row_sum.fill(0.0)

        # group parameters for compute_one_n_block
        basic_params = SimpleNamespace(
            m_block=m_block,
            n_block=n_block,
            mQ=mQ,
            mK=mK,
            batch_size=batch_size,
            num_head=num_head,
        )
        mma_params = SimpleNamespace(
            thr_mma=thr_mma,
            tiled_mma=tiled_mma,
            tSrQ=tSrQ,
            tSrK=tSrK,
            tOrVt=tOrVt,
            acc_O=acc_O,
        )
        gmem_copy_params = SimpleNamespace(
            gmem_tiled_copy_QKV=gmem_tiled_copy_QKV,
            tKVcKV=tKVcKV,
            tKgK=tKgK,
            tKsK=tKsK,
            tVgV=tVgV,
            tVsV=tVsV,
            tKVpKV=tKVpKV,
        )
        smem_copy_params = SimpleNamespace(
            smem_tiled_copy_Q=smem_tiled_copy_Q,
            smem_tiled_copy_K=smem_tiled_copy_K,
            smem_tiled_copy_V=smem_tiled_copy_V,
            tSsQ=tSsQ,
            tSrQ_copy_view=tSrQ_copy_view,
            tSsK=tSsK,
            tSrK_copy_view=tSrK_copy_view,
            tOsVt=tOsVt,
            tOrVt_copy_view=tOrVt_copy_view,
        )
        softmax_params = SimpleNamespace(
            row_max=row_max,
            row_sum=row_sum,
            softmax_scale_log2=softmax_scale_log2,
        )

        # Start processing of the first n-block.
        # For performance reason, we separate out two kinds of iterations:
        # those that need masking on S, and those that don't.
        # We need masking on S for the very last block when K and V has length not multiple of n_block_size.
        # We also need masking on S if it's causal, for the last ceil_div(m_block_size, n_block_size) blocks.
        # We will have at least 1 "masking" iteration.
        mask_steps = 1
        if cutlass.const_expr(self._is_causal):
            mask_steps = cute.ceil_div(self._m_block_size, self._n_block_size)

        for n_tile in cutlass.range_constexpr(mask_steps):
            n_block = n_block_max - n_tile - 1
            basic_params.n_block = n_block
            if cutlass.const_expr(self._is_causal):
                if n_block >= 0:
                    self.compute_one_n_block(
                        basic_params,
                        mma_params,
                        gmem_copy_params,
                        smem_copy_params,
                        softmax_params,
                        is_first_n_block=(n_tile == 0),
                        in_mask_steps=True,
                    )
            else:
                self.compute_one_n_block(
                    basic_params,
                    mma_params,
                    gmem_copy_params,
                    smem_copy_params,
                    softmax_params,
                    is_first_n_block=True,
                    in_mask_steps=True,
                )

        # Start async loads of rest k-tiles in reverse order, no k-residue handling needed
        for n_tile in range(mask_steps, n_block_max, 1):
            n_block = n_block_max - n_tile - 1
            basic_params.n_block = n_block
            self.compute_one_n_block(
                basic_params,
                mma_params,
                gmem_copy_params,
                smem_copy_params,
                softmax_params,
                is_first_n_block=False,
                in_mask_steps=False,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        # Epilogue
        # ///////////////////////////////////////////////////////////////////////////////
        # normalize acc_O by row_sum and calculate the lse
        self.normalize_softmax(acc_O, row_sum)
        # store acc_O
        rO = cute.make_fragment_like(acc_O, self._dtype)
        rO.store(acc_O.load().to(self._dtype))
        # reuse sQ's data iterator
        sO = cute.make_tensor(sQ.iterator, sO_layout)

        # smem copy atom for O
        smem_copy_atom_O = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self._dtype
        )
        # tiled copy atom for O
        smem_tiled_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma)
        smem_thr_copy_O = smem_tiled_copy_O.get_slice(tidx)
        taccOrO = smem_thr_copy_O.retile(rO)
        taccOsO = smem_thr_copy_O.partition_D(sO)
        # copy acc O from rmem to smem with the smem copy atom
        cute.copy(
            smem_copy_atom_O,
            taccOrO,
            taccOsO,
        )
        gO = cute.local_tile(
            mO[batch_size, None, num_head, None],
            (self._m_block_size, self._head_dim_padded),
            (m_block, 0),
        )

        gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
        tOsO = gmem_thr_copy_O.partition_S(sO)
        tOgO = gmem_thr_copy_O.partition_D(gO)
        tOrO = cute.make_fragment_like(tOgO, self._dtype)
        # sync before all smem stores are done.
        self.cta_sync_barrier.arrive_and_wait()
        # load acc O from smem to rmem for wider vectorization
        cute.copy(
            gmem_tiled_copy_O,
            tOsO,
            tOrO,
        )
        mcO = cute.make_identity_tensor(mO.layout.shape)
        cO = cute.local_tile(
            mcO[batch_size, None, num_head, None],
            (self._m_block_size, self._head_dim_padded),
            (m_block, 0),
        )
        tOcO = gmem_thr_copy_O.partition_D(cO)
        tOpO = cute.make_rmem_tensor(
            cute.make_layout(
                (tOgO.shape[0][1], tOgO.shape[1], tOgO.shape[2]),
                stride=(tOgO.shape[2], 0, 1),
            ),
            cutlass.Boolean,
        )
        for rest_v in cutlass.range_constexpr(tOpO.shape[0]):
            for rest_n in cutlass.range_constexpr(cute.size(tOpO.shape[2])):
                tOpO[rest_v, 0, rest_n] = cute.elem_less(
                    tOcO[(0, rest_v), 0, rest_n][3], mO.layout.shape[3]
                )
        # copy acc O from rmem to gmem
        for rest_m in cutlass.range_constexpr(cute.size(tOpO.shape[1])):
            if cute.elem_less(tOcO[0, rest_m, 0][1], mO.layout.shape[1]):
                cute.copy(
                    gmem_tiled_copy_O,
                    tOrO[None, rest_m, None],
                    tOgO[None, rest_m, None],
                    pred=tOpO[None, rest_m, None],
                )

    @cute.jit
    def compute_one_n_block(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        gmem_copy_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        is_first_n_block: cutlass.Constexpr,
        in_mask_steps: cutlass.Constexpr,
    ):
        """Compute one n_block of S/O.

        This function provides different variants for processing the first n block versus subsequent blocks,
        as well as variants for handling masked and unmasked steps.

        :param basic_params: basic parameters
        :type basic_params: SimpleNamespace
        :param mma_params: mma parameters
        :type mma_params: SimpleNamespace
        :param gmem_copy_params: gmem copy parameters
        :type gmem_copy_params: SimpleNamespace
        :param smem_copy_params: smem copy parameters
        :type smem_copy_params: SimpleNamespace
        :param softmax_params: softmax parameters
        :type softmax_params: SimpleNamespace
        :param is_first_n_block: is first n block
        :type is_first_n_block: cutlass.Constexpr
        """
        acc_shape_S = mma_params.thr_mma.partition_shape_C(
            (self._m_block_size, self._n_block_size)
        )
        acc_S = cute.make_rmem_tensor(acc_shape_S, cutlass.Float32)
        acc_S.fill(0.0)

        # wait for smem tile QK before mma calculation for S
        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()
        # load smem tile V for O, special process for the first tile to avoid loading nan.
        # The `if` here is a constexpr, won't be generated in the IR.
        if is_first_n_block:
            for n in cutlass.range_constexpr(cute.size(gmem_copy_params.tVsV.shape[1])):
                if cute.elem_less(
                    gmem_copy_params.tKVcKV[0, n, 0][1],
                    basic_params.mK.layout.shape[1],
                ):
                    cute.copy(
                        gmem_copy_params.gmem_tiled_copy_QKV,
                        gmem_copy_params.tVgV[None, n, None, basic_params.n_block],
                        gmem_copy_params.tVsV[None, n, None],
                        pred=gmem_copy_params.tKVpKV[None, n, None],
                    )
                else:
                    gmem_copy_params.tVsV[None, n, None].fill(0.0)
        else:
            cute.copy(
                gmem_copy_params.gmem_tiled_copy_QKV,
                gmem_copy_params.tVgV[None, None, None, basic_params.n_block],
                gmem_copy_params.tVsV,
                pred=gmem_copy_params.tKVpKV,
            )

        cute.arch.cp_async_commit_group()
        # ///////////////////////////////////////////////////////////////////////////////
        # S gemm calculation
        # ///////////////////////////////////////////////////////////////////////////////
        # load first QK k-block from smem to rmem for mma
        cute.copy(
            smem_copy_params.smem_tiled_copy_Q,
            smem_copy_params.tSsQ[None, None, 0],
            smem_copy_params.tSrQ_copy_view[None, None, 0],
        )
        cute.copy(
            smem_copy_params.smem_tiled_copy_K,
            smem_copy_params.tSsK[None, None, 0],
            smem_copy_params.tSrK_copy_view[None, None, 0],
        )
        # mma for S
        for k in cutlass.range_constexpr(cute.size(smem_copy_params.tSsQ.shape[2])):
            # load next QK k-block from smem to rmem for mma
            k_next = (k + 1) % cute.size(smem_copy_params.tSsQ.shape[2])
            cute.copy(
                smem_copy_params.smem_tiled_copy_Q,
                smem_copy_params.tSsQ[None, None, k_next],
                smem_copy_params.tSrQ_copy_view[None, None, k_next],
            )
            cute.copy(
                smem_copy_params.smem_tiled_copy_K,
                smem_copy_params.tSsK[None, None, k_next],
                smem_copy_params.tSrK_copy_view[None, None, k_next],
            )
            cute.gemm(
                mma_params.tiled_mma,
                acc_S,
                mma_params.tSrQ[None, None, k],
                mma_params.tSrK[None, None, k],
                acc_S,
            )

        # wait for smem tile V for O
        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()

        if basic_params.n_block > 0:
            cute.copy(
                gmem_copy_params.gmem_tiled_copy_QKV,
                gmem_copy_params.tKgK[None, None, None, basic_params.n_block - 1],
                gmem_copy_params.tKsK,
                pred=gmem_copy_params.tKVpKV,
            )
            cute.arch.cp_async_commit_group()
        # ///////////////////////////////////////////////////////////////////////////////
        # online softmax
        # ///////////////////////////////////////////////////////////////////////////////
        self.softmax_rescale_O(
            basic_params,
            mma_params,
            softmax_params,
            acc_S,
            is_first_n_block,
            in_mask_steps,
        )

        rP = cute.make_fragment_like(acc_S, self._dtype)
        rP.store(acc_S.load().to(self._dtype))
        # ///////////////////////////////////////////////////////////////////////////////
        # O gemm calculation
        # ///////////////////////////////////////////////////////////////////////////////
        # Convert layout of acc_S to gemm O accept layout.
        # Due to the mma instruction shape is 16x8x16, we need to convert from (4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        # (4, MMA_M, MMA_N) -> (4, MMA_M, (2, MMA_N / 2))
        rP_layout_divided = cute.logical_divide(rP.layout, (None, None, 2))
        rP_mma_view = cute.make_layout(
            (
                (rP_layout_divided.shape[0], rP_layout_divided.shape[2][0]),
                rP_layout_divided.shape[1],
                rP_layout_divided.shape[2][1],
            ),
            stride=(
                (rP_layout_divided.stride[0], rP_layout_divided.stride[2][0]),
                rP_layout_divided.stride[1],
                rP_layout_divided.stride[2][1],
            ),
        )
        tOrS = cute.make_tensor(rP.iterator, rP_mma_view)

        # load first V k-block from smem to rmem for mma
        cute.copy(
            smem_copy_params.smem_tiled_copy_V,
            smem_copy_params.tOsVt[None, None, 0],
            smem_copy_params.tOrVt_copy_view[None, None, 0],
        )
        # mma for O
        for k in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
            # load next V k-block from smem to rmem for mma
            k_next = (k + 1) % cute.size(tOrS.shape[2])
            cute.copy(
                smem_copy_params.smem_tiled_copy_V,
                smem_copy_params.tOsVt[None, None, k_next],
                smem_copy_params.tOrVt_copy_view[None, None, k_next],
            )
            cute.gemm(
                mma_params.tiled_mma,
                mma_params.acc_O,
                tOrS[None, None, k],
                mma_params.tOrVt[None, None, k],
                mma_params.acc_O,
            )

    @cute.jit
    def softmax_rescale_O(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        acc_S: cute.Tensor,
        is_first_n_block: cutlass.Constexpr,
        in_mask_steps: cutlass.Constexpr,
    ):
        """Apply online softmax and rescale acc_O.

        This function provides different variants for processing the first n block versus subsequent blocks,
        as well as variants for handling masked and unmasked steps.

        :param basic_params: basic parameters
        :type basic_params: SimpleNamespace
        :param mma_params: mma parameters
        :type mma_params: SimpleNamespace
        :param softmax_params: softmax parameters
        :type softmax_params: SimpleNamespace
        :param acc_S: acc_S tensor
        :type acc_S: cute.Tensor
        :param is_first_n_block: is first n_block
        :type is_first_n_block: cutlass.Constexpr
        :param in_mask_steps: in mask steps
        :type in_mask_steps: cutlass.Constexpr
        """
        # Change acc_S to M,N layout view.
        acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
        acc_O_mn = self._make_acc_tensor_mn_view(mma_params.acc_O)
        row_max_prev = None
        # if it is not the first tile, load the row r of previous row_max and compare with row_max_cur_row.
        if cutlass.const_expr(not is_first_n_block):
            row_max_prev = cute.make_fragment_like(
                softmax_params.row_max, cutlass.Float32
            )
            cute.basic_copy(softmax_params.row_max, row_max_prev)
        # if it is the first tile, create a mask for residual of S to -inf for softmax.
        tScS_mn = None
        if cutlass.const_expr(in_mask_steps):
            mcS = cute.make_identity_tensor(
                (
                    basic_params.mQ.shape[0],
                    basic_params.mQ.shape[1],
                    basic_params.mQ.shape[2],
                    basic_params.mK.shape[1],
                )
            )
            cS = cute.local_tile(
                mcS[basic_params.batch_size, None, basic_params.num_head, None],
                (self._m_block_size, self._n_block_size),
                (basic_params.m_block, basic_params.n_block),
            )
            tScS = mma_params.thr_mma.partition_C(cS)
            tScS_mn = self._make_acc_tensor_mn_view(tScS)

        # Each iteration processes one row of acc_S
        for r in cutlass.range_constexpr(cute.size(softmax_params.row_max)):
            # mask residual of S with -inf
            if cutlass.const_expr(in_mask_steps):
                if cutlass.const_expr(not self._is_causal):
                    # traverse column index.
                    for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                        if cute.elem_less(
                            basic_params.mK.shape[1], tScS_mn[0, c][3] + 1
                        ):
                            acc_S_mn[r, c] = -cutlass.Float32.inf
                else:
                    # get the column index limit based on current row. Only consider the row index, so the column index sets to 0.
                    col_idx_limit = cutlass.min(
                        tScS_mn[r, 0][1] + 1, basic_params.mK.shape[1]
                    )
                    # traverse column index.
                    for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                        # only consider the column index, so the row index sets to 0.
                        if cute.elem_less(col_idx_limit, tScS_mn[0, c][3] + 1):
                            acc_S_mn[r, c] = -cutlass.Float32.inf

            # (n_block_size)
            acc_S_row = acc_S_mn[r, None].load()
            # row_max_cur_row => f32
            row_max_cur_row = acc_S_row.reduce(
                cute.ReductionOp.MAX, -cutlass.Float32.inf, 0
            )
            # quad reduction for row_max
            row_max_cur_row = self._threadquad_reduce_max(row_max_cur_row)
            row_max_prev_row = None
            # if it is not the first tile, load the row r of previous row_max and compare with row_max_cur_row.
            if cutlass.const_expr(not is_first_n_block):
                row_max_prev_row = row_max_prev[r]
                row_max_cur_row = cute.arch.fmax(row_max_prev_row, row_max_cur_row)
            if cutlass.const_expr(self._is_causal):
                row_max_cur_row = (
                    0.0 if row_max_cur_row == -cutlass.Float32.inf else row_max_cur_row
                )

            # compute exp(x - max) using exp2(x * log_2(e) - max * log_2(e))
            acc_S_row_exp = cute.math.exp2(
                acc_S_row * softmax_params.softmax_scale_log2
                - row_max_cur_row * softmax_params.softmax_scale_log2,
                fastmath=True,
            )
            # acc_S_row_sum => f32
            acc_S_row_sum = acc_S_row_exp.reduce(
                cute.ReductionOp.ADD, cutlass.Float32.zero, 0
            )
            # if it is not the first tile, load the row r of previous row_max and minus row_max_cur_row to update row_sum.
            if cutlass.const_expr(not is_first_n_block):
                prev_minus_cur_exp = cute.math.exp2(
                    row_max_prev_row * softmax_params.softmax_scale_log2
                    - row_max_cur_row * softmax_params.softmax_scale_log2,
                    fastmath=True,
                )
                acc_S_row_sum = (
                    acc_S_row_sum + softmax_params.row_sum[r] * prev_minus_cur_exp
                )
                acc_O_mn[r, None] = acc_O_mn[r, None].load() * prev_minus_cur_exp
            # update row_max, row_sum and acc_S
            softmax_params.row_max[r] = row_max_cur_row
            softmax_params.row_sum[r] = acc_S_row_sum
            acc_S_mn[r, None] = acc_S_row_exp

    @cute.jit
    def normalize_softmax(
        self,
        acc_O: cute.Tensor,
        row_sum: cute.Tensor,
    ):
        """Normalize acc_O by row_sum.

        :param acc_O: input tensor
        :type acc_O: cute.Tensor
        :param row_sum: row_sum tensor
        :type row_sum: cute.Tensor
        """
        # do quad reduction for row_sum.
        acc_O_mn = self._make_acc_tensor_mn_view(acc_O)
        for r in cutlass.range_constexpr(cute.size(row_sum)):
            row_sum[r] = self._threadquad_reduce_sum(row_sum[r])
            # if row_sum is zero or nan, set acc_O_mn_row to 1.0
            acc_O_mn_row_is_zero_or_nan = row_sum[r] == 0.0 or row_sum[r] != row_sum[r]

            scale = (
                1.0 if acc_O_mn_row_is_zero_or_nan else cute.arch.rcp_approx(row_sum[r])
            )

            acc_O_mn[r, None] = acc_O_mn[r, None].load() * scale

    def _make_acc_tensor_mn_view(self, acc: cute.Tensor) -> cute.Tensor:
        """make acc tensor as mn layout view

        :param acc: input tensor
        :type acc: cute.Tensor
        :return: acc tensor mn layout view
        :rtype: cute.Tensor
        """
        acc_layout_col_major = cute.make_layout(acc.layout.shape)
        acc_layout_mn = cute.make_layout(
            (
                (
                    acc_layout_col_major.shape[0][1],
                    acc_layout_col_major.shape[1],
                ),  # MMA_M
                (
                    acc_layout_col_major.shape[0][0],
                    acc_layout_col_major.shape[2],
                ),  # MMA_N
            ),
            stride=(
                (
                    acc_layout_col_major.stride[0][1],
                    acc_layout_col_major.stride[1],
                ),  # MMA_M
                (
                    acc_layout_col_major.stride[0][0],
                    acc_layout_col_major.stride[2],
                ),  # MMA_N
            ),
        )
        acc_layout_mn = cute.composition(acc.layout, acc_layout_mn)
        return cute.make_tensor(acc.iterator, acc_layout_mn)

    def _threadquad_reduce(self, val: cutlass.Float32, op: Callable) -> cutlass.Float32:
        """thread quad reduction

        :param val: register value
        :type val: cutlass.Float32
        :param op: binary operator
        :type op: Callable
        :return: reduced value
        :rtype: cutlass.Float32
        """
        val = op(
            val,
            cute.arch.shuffle_sync_bfly(val, offset=2, mask=-1, mask_and_clamp=31),
        )
        val = op(
            val,
            cute.arch.shuffle_sync_bfly(val, offset=1, mask=-1, mask_and_clamp=31),
        )
        return val

    def _threadquad_reduce_max(self, val: cutlass.Float32) -> cutlass.Float32:
        """thread quad reduction max

        :param val: register value
        :type val: cutlass.Float32
        :return: max value
        :rtype: cutlass.Float32
        """
        return self._threadquad_reduce(val, lambda x, y: cute.arch.fmax(x, y))

    def _threadquad_reduce_sum(self, val: cutlass.Float32) -> cutlass.Float32:
        """thread quad reduction sum

        :param val: register value
        :type val: cutlass.Float32
        :return: sum value
        :rtype: cutlass.Float32
        """
        return self._threadquad_reduce(val, lambda x, y: x + y)


class SparkReweightForwardSm80(FlashAttentionForwardAmpere):
    """Fused Spark attention mainloop for Ampere SM80.

    This is the SM80 counterpart of :class:`SparkReweightForwardSm120`.
    Query tiles stay resident in shared memory while one CTA builds its route
    mask, folds skipped virtual summaries into the online softmax, and streams
    the selected physical K/V blocks into the same FP32 output accumulator.
    No dense route tensor, exact-output tensor, or post-attention merge is
    materialized.
    """

    def __init__(
        self,
        *,
        external_route: bool = True,
        packed_external_route: bool = True,
        skip_external_route_qk: bool = True,
        hybrid_route: bool = False,
        export_route: bool = False,
        force_local_blocks: bool = True,
        fused_topk_ratio: float = 0.0,
        runtime_fused_topk_ratio: bool = True,
        prefetch_summary: bool = True,
        skip_final_tile_barrier: bool = True,
        **_kwargs,
    ):
        super().__init__(128, m_block_size=64, n_block_size=64,
                         num_threads=128, is_causal=False)
        if skip_external_route_qk and not packed_external_route:
            raise ValueError(
                "route-QK-free specialization requires packed external routing"
            )
        if packed_external_route and not external_route:
            raise ValueError("packed external routing requires external_route=True")
        if hybrid_route or export_route or (
            external_route and not (
                packed_external_route and skip_external_route_qk
            )
        ):
            raise NotImplementedError(
                "SM80/SM89 fused Spark implements only route-QK-free packed external routing"
            )
        self.external_route = external_route
        self.packed_external_route = packed_external_route
        self.skip_external_route_qk = skip_external_route_qk
        if fused_topk_ratio < 0.0 or fused_topk_ratio > 1.0:
            raise ValueError("fused_topk_ratio must be in [0, 1]")
        self.fused_topk_route = fused_topk_ratio > 0.0
        self.fused_topk_numerator = round(fused_topk_ratio * 10000)
        self.runtime_fused_topk_ratio = runtime_fused_topk_ratio
        from .local_blocks import normalize_local_block_radius
        explicit_local_radius = type(force_local_blocks) is int
        self.local_block_radius = normalize_local_block_radius(force_local_blocks)
        if external_route and not explicit_local_radius:
            self.local_block_radius = -1
        self.prefetch_summary = prefetch_summary
        self.skip_final_tile_barrier = skip_final_tile_barrier

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        o: cute.Tensor,
        kc: cute.Tensor,
        av: cute.Tensor,
        threshold: cute.Tensor,
        route_mask: cute.Tensor,
        lse: cute.Tensor,
        ak: cute.Tensor,
        lm: cute.Tensor,
        mapping: cute.Tensor,
        q_block_start: cutlass.Int32,
        q_block_count: cutlass.Int32,
        parent_start: cutlass.Int32,
        head_start: cutlass.Int32,
        head_count: cutlass.Int32,
        softmax_scale: cutlass.Float32,
        sink_start_block: cutlass.Int32,
        sink_end_block: cutlass.Int32,
        fused_topk_numerator: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(
            q.element_type != cutlass.BFloat16
            or k.element_type != cutlass.BFloat16
            or v.element_type != cutlass.BFloat16
            or o.element_type != cutlass.BFloat16
        ):
            raise TypeError("SM80/SM89 fused Spark requires BF16 Q/K/V/O")
        self._dtype = q.element_type
        smem_atom = cute.make_composed_layout(
            cute.make_swizzle(3, 3, 3),
            0,
            cute.make_layout((8, 64), stride=(64, 1)),
        )
        sQ_layout = cute.tile_to_shape(smem_atom, (64, 128), (0, 1))
        # Two K/V stages let the exact-block mainloop overlap cp.async with
        # tensor-core work, matching the producer/consumer structure used by
        # the SM120 implementation.
        sKV_layout = cute.tile_to_shape(smem_atom, (64, 128, 2), (0, 1, 2))

        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sQ_layout)], 1024
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024
            ]
            qmean: cute.struct.Align[cute.struct.MemRange[self._dtype, 128], 16]
            route: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 64], 16]
            skipped: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 1], 4]
            exact_count: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 1], 4
            ]
            exact_indices: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 64], 16
            ]

        atom_async = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self._dtype,
            num_bits_per_copy=128,
        )
        copy_layout = cute.make_layout((16, 8), stride=(8, 1))
        value_layout = cute.make_layout((1, 8))
        gmem_copy = cute.make_tiled_copy_tv(atom_async, copy_layout, value_layout)
        store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self._dtype, num_bits_per_copy=128
        )
        gmem_store = cute.make_tiled_copy_tv(store_atom, copy_layout, value_layout)
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
            (4, 1, 1),
            permutation_mnk=(64, 16, 16),
        )
        self.fused_kernel(
            q, k, v, o, kc, av, threshold, route_mask, lse, ak, lm, mapping,
            q_block_start, parent_start, head_start, softmax_scale,
            sink_start_block, sink_end_block, fused_topk_numerator,
            sQ_layout, sKV_layout,
            gmem_copy, gmem_store, tiled_mma, SharedStorage,
        ).launch(
            grid=(q_block_count, head_count, q.shape[0]),
            block=(128, 1, 1),
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def fused_kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mKC: cute.Tensor,
        mAV: cute.Tensor,
        mThreshold: cute.Tensor,
        mRouteMask: cute.Tensor,
        mLSE: cute.Tensor,
        mAK: cute.Tensor,
        mLM: cute.Tensor,
        mMap: cute.Tensor,
        q_block_start: cutlass.Int32,
        parent_start: cutlass.Int32,
        head_start: cutlass.Int32,
        softmax_scale: cutlass.Float32,
        sink_start_block: cutlass.Int32,
        sink_end_block: cutlass.Int32,
        fused_topk_numerator: cutlass.Int32,
        sQ_layout: cute.ComposedLayout,
        sKV_layout: cute.ComposedLayout,
        gmem_copy: cute.TiledCopy,
        gmem_store: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        q_local, head_local, batch = cute.arch.block_idx()
        q_block = cute.arch.make_warp_uniform(q_local + q_block_start)
        head = cute.arch.make_warp_uniform(head_local + head_start)
        parent = cutlass.Int32(mMap[q_block]) - parent_start
        q_start = q_block * 64
        tokens = mK.shape[1]
        blocks = mKC.shape[1]
        q_len = cutlass.Int32(64)
        if q_start + 64 > tokens:
            q_len = cutlass.Int32(tokens - q_start)
        threshold = cutlass.Float32(mThreshold[batch, q_block, head])
        scale_log2 = softmax_scale * 1.4426950408889634

        storage = cutlass.utils.SmemAllocator().allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sKV_layout)
        sV = storage.sV.get_tensor(sKV_layout)
        qmean_ptr = storage.qmean.data_ptr()
        # Match SM120's shared_log_mass path: each route column stores either
        # -inf (exact/invalid) or its pre-scaled approximate log-mass. This
        # avoids reloading LM for every query-row score fragment.
        route_ptr = cute.recast_ptr(
            storage.route.data_ptr(), dtype=cutlass.Float32
        )
        skipped_ptr = storage.skipped.data_ptr()
        exact_count_ptr = storage.exact_count.data_ptr()
        exact_indices_ptr = storage.exact_indices.data_ptr()
        q_scratch_f32_ptr = cute.recast_ptr(
            storage.sQ.data_ptr(), dtype=cutlass.Float32
        )
        fused_route_scores_ptr = q_scratch_f32_ptr
        sVt = cute.composition(
            sV, cute.make_layout((128, 64, 2), stride=(64, 1, 8192))
        )

        gQ = cute.local_tile(
            mQ[batch, None, head, None], (64, 128), (q_block, 0)
        )
        gK = cute.local_tile(
            mK[batch, None, head, None], (64, 128), (None, 0)
        )
        gV = cute.local_tile(
            mV[batch, None, head, None], (64, 128), (None, 0)
        )
        # AK/AV are flattened as [batch * parent_chunk, block, local_head, D].
        summary_batch = batch * mLM.shape[1] + parent
        gAK = cute.local_tile(
            mAK[summary_batch, None, head_local, None],
            (64, 128), (None, 0),
        )
        gAV = cute.local_tile(
            mAV[summary_batch, None, head_local, None],
            (64, 128), (None, 0),
        )

        thr_copy = gmem_copy.get_slice(tidx)
        tQgQ, tQsQ = thr_copy.partition_S(gQ), thr_copy.partition_D(sQ)
        tKgK, tKsK = thr_copy.partition_S(gK), thr_copy.partition_D(sK)
        tVgV, tVsV = thr_copy.partition_S(gV), thr_copy.partition_D(sV)
        tAKgAK = thr_copy.partition_S(gAK)
        tAVgAV = thr_copy.partition_S(gAV)
        cK = cute.local_tile(
            cute.make_identity_tensor(mK.layout.shape)[batch, None, head, None],
            (64, 128), (None, 0),
        )
        cQ = cute.local_tile(
            cute.make_identity_tensor(mQ.layout.shape)[batch, None, head, None],
            (64, 128), (q_block, 0),
        )
        tQcQ = thr_copy.partition_S(cQ)
        cAK = cute.local_tile(
            cute.make_identity_tensor(mAK.layout.shape)[
                summary_batch, None, head_local, None
            ],
            (64, 128), (None, 0),
        )
        tKcK = thr_copy.partition_S(cK)
        tAKcAK = thr_copy.partition_S(cAK)

        # Q has a complete 128-wide head dimension. Predicate the row groups so
        # a standalone full-length call cannot read past a partial final tile.
        for row_part in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            if cute.elem_less(tQcQ[0, row_part, 0][1], tokens):
                cute.copy(gmem_copy, tQgQ[None, row_part, None],
                          tQsQ[None, row_part, None])
            else:
                tQsQ[None, row_part, None].fill(0)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()

        # External packed routing already contains the exact mask, so its
        # specialization does not build a query centroid or execute route QK.
        if cutlass.const_expr(not self.skip_external_route_qk):
            qsum = cutlass.Float32.zero
            for row in cutlass.range_constexpr(64):
                qsum += cutlass.Float32(sQ[row, tidx])
            (qmean_ptr + tidx).store(qsum / cutlass.Float32(q_len))
            self.cta_sync_barrier.arrive_and_wait()

        thr_mma = tiled_mma.get_slice(tidx)
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK[None, None, 0]))
        tOrV = thr_mma.make_fragment_B(thr_mma.partition_B(sVt[None, None, 0]))
        acc_O = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((64, 128)), cutlass.Float32
        )
        acc_O.fill(0.0)
        row_max = cute.make_rmem_tensor(
            (acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32
        )
        row_sum = cute.make_rmem_tensor_like(row_max, cutlass.Float32)
        row_max.fill(-cutlass.Float32.inf)
        row_sum.fill(0.0)

        copy_q = cute.make_tiled_copy_A(
            cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                self._dtype,
            ), tiled_mma,
        )
        copy_k = cute.make_tiled_copy_B(
            cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                self._dtype,
            ), tiled_mma,
        )
        copy_v = cute.make_tiled_copy_B(
            cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
                self._dtype,
            ), tiled_mma,
        )
        q_slice, k_slice, v_slice = (
            copy_q.get_slice(tidx), copy_k.get_slice(tidx),
            copy_v.get_slice(tidx),
        )
        tSsQ = q_slice.partition_S(sQ)
        tSsK = k_slice.partition_S(sK)
        tSsV = v_slice.partition_S(sVt)
        tSrQ_view = q_slice.retile(tSrQ)
        tSrK_view = k_slice.retile(tSrK)
        tOrV_view = v_slice.retile(tOrV)
        cute.copy(copy_q, tSsQ, tSrQ_view)

        local_coords = cute.make_identity_tensor((64, 64))
        score_coords = thr_mma.partition_C(local_coords)
        num_groups = cute.ceil_div(blocks, 64)
        candidate_blocks = sink_start_block
        if candidate_blocks > blocks:
            candidate_blocks = cutlass.Int32(blocks)

        # Port of SM120's CTA-local fixed-budget Top-K route.  Computing the
        # mean-Q dot centroid directly is algebraically identical to reducing
        # all 64 rows of Q.Kc, but avoids a redundant tensor-core GEMM on SM80.
        # The score array aliases sQ only after the Q fragment is register
        # resident, so this adds no shared-memory footprint.
        if cutlass.const_expr(self.fused_topk_route):
            for route_group in range(0, num_groups, 1):
                group_start = route_group * 64
                for item in cutlass.range_constexpr(16):
                    column = warp_idx + item * 4
                    block = group_start + column
                    route_score = cutlass.Float32.zero
                    if block < candidate_blocks:
                        for part in cutlass.range_constexpr(4):
                            dim = lane + part * 32
                            route_score += (
                                cutlass.Float32((qmean_ptr + dim).load())
                                * cutlass.Float32(mKC[batch, block, head, dim])
                            )
                        route_score = self._warp_reduce_sum(route_score)
                    if lane == 0 and block < candidate_blocks:
                        (fused_route_scores_ptr + block).store(
                            route_score * scale_log2
                        )
            self.cta_sync_barrier.arrive_and_wait()

            topk_numerator = (
                fused_topk_numerator
                if cutlass.const_expr(self.runtime_fused_topk_ratio)
                else cutlass.Int32(self.fused_topk_numerator)
            )
            target_topk = (
                candidate_blocks * topk_numerator
                + cutlass.Int32(5000)
            ) // cutlass.Int32(10000)
            if target_topk < cutlass.Int32(1):
                target_topk = cutlass.Int32(1)
            if target_topk > candidate_blocks:
                target_topk = candidate_blocks

            # SM120 repeatedly extracts one maximum. On Ampere, use an exact
            # binary radix selector instead: it finds the K-th ordered FP32
            # key, resolves boundary ties by lower block index, and marks
            # precisely K scores. This removes O(K) CTA barriers while keeping
            # the same deterministic total order. All four warps scan each
            # bucket; the earlier one-warp version left 75% of the CTA idle.
            # Per-warp counts use route scratch that is overwritten below.
            prefix = cutlass.Uint32(0)
            prefix_mask = cutlass.Uint32(0)
            remaining_rank = cutlass.Int32(target_topk)
            active_count = cutlass.Int32(candidate_blocks)
            tie_index = cutlass.Int32(-1)
            for shift in cutlass.range_constexpr(31, -1, -1):
                bit = cutlass.Uint32(1 << shift)
                if active_count > 1:
                    local_ones = cutlass.Int32(0)
                    index = cutlass.Int32(tidx)
                    while index < candidate_blocks:
                        value = cutlass.Float32(
                            (fused_route_scores_ptr + index).load()
                        )
                        bits = value.bitcast(cutlass.Uint32)
                        ordered = bits ^ cutlass.Uint32(0x80000000)
                        if (bits & cutlass.Uint32(0x80000000)) != 0:
                            ordered = bits ^ cutlass.Uint32(0xFFFFFFFF)
                        if (
                            (ordered & prefix_mask) == prefix
                            and (ordered & bit) != 0
                        ):
                            local_ones += 1
                        index += cutlass.Int32(128)
                    for offset in (16, 8, 4, 2, 1):
                        local_ones += cute.arch.shuffle_sync_down(
                            local_ones, offset=offset,
                            mask=-1, mask_and_clamp=31,
                        )
                    if lane == 0:
                        (route_ptr + warp_idx).store(
                            cutlass.Float32(local_ones)
                        )
                    self.cta_sync_barrier.arrive_and_wait()
                    if tidx == 0:
                        total_ones = cutlass.Int32(0)
                        for owner in cutlass.range_constexpr(4):
                            total_ones += cutlass.Int32(
                                (route_ptr + owner).load()
                            )
                        route_ptr.store(cutlass.Float32(total_ones))
                    self.cta_sync_barrier.arrive_and_wait()
                    ones = cutlass.Int32(route_ptr.load())
                    if remaining_rank <= ones:
                        prefix |= bit
                        active_count = ones
                    else:
                        remaining_rank -= ones
                        active_count -= ones
                    prefix_mask |= bit

                    # Once the selected radix bucket is unique, recover its
                    # complete ordered key once and skip the remaining scans.
                    if active_count == 1:
                        unique_index = cutlass.Int32(-1)
                        index = cutlass.Int32(tidx)
                        while index < candidate_blocks:
                            value = cutlass.Float32(
                                (fused_route_scores_ptr + index).load()
                            )
                            bits = value.bitcast(cutlass.Uint32)
                            ordered = bits ^ cutlass.Uint32(0x80000000)
                            if (bits & cutlass.Uint32(0x80000000)) != 0:
                                ordered = bits ^ cutlass.Uint32(0xFFFFFFFF)
                            if (ordered & prefix_mask) == prefix:
                                unique_index = index
                            index += cutlass.Int32(128)
                        for offset in (16, 8, 4, 2, 1):
                            peer_index = cute.arch.shuffle_sync_down(
                                unique_index, offset=offset,
                                mask=-1, mask_and_clamp=31,
                            )
                            if peer_index > unique_index:
                                unique_index = peer_index
                        if lane == 0:
                            (exact_indices_ptr + warp_idx).store(unique_index)
                        self.cta_sync_barrier.arrive_and_wait()
                        if tidx == 0:
                            cta_index = cutlass.Int32(-1)
                            for owner in cutlass.range_constexpr(4):
                                peer_index = cutlass.Int32(
                                    (exact_indices_ptr + owner).load()
                                )
                                if peer_index > cta_index:
                                    cta_index = peer_index
                            exact_indices_ptr.store(cta_index)
                        self.cta_sync_barrier.arrive_and_wait()
                        tie_index = cutlass.Int32(exact_indices_ptr.load())
                        value = cutlass.Float32(
                            (fused_route_scores_ptr + tie_index).load()
                        )
                        bits = value.bitcast(cutlass.Uint32)
                        prefix = bits ^ cutlass.Uint32(0x80000000)
                        if (bits & cutlass.Uint32(0x80000000)) != 0:
                            prefix = bits ^ cutlass.Uint32(0xFFFFFFFF)
                        prefix_mask = cutlass.Uint32(0xFFFFFFFF)

            # remaining_rank is the one-based rank within scores whose
            # ordered bits equal prefix. Locate that tie in block order. This
            # cold path handles genuinely equal FP32 route scores.
            if tidx == 0 and tie_index < 0:
                equal_seen = cutlass.Int32(0)
                index = cutlass.Int32(0)
                while index < candidate_blocks:
                    value = cutlass.Float32(
                        (fused_route_scores_ptr + index).load()
                    )
                    bits = value.bitcast(cutlass.Uint32)
                    ordered = bits ^ cutlass.Uint32(0x80000000)
                    if (bits & cutlass.Uint32(0x80000000)) != 0:
                        ordered = bits ^ cutlass.Uint32(0xFFFFFFFF)
                    if ordered == prefix:
                        equal_seen += 1
                        if equal_seen == remaining_rank:
                            tie_index = index
                    index += 1
                exact_indices_ptr.store(tie_index)
            self.cta_sync_barrier.arrive_and_wait()
            tie_index = cutlass.Int32(exact_indices_ptr.load())

            index = cutlass.Int32(tidx)
            while index < candidate_blocks:
                value = cutlass.Float32(
                    (fused_route_scores_ptr + index).load()
                )
                bits = value.bitcast(cutlass.Uint32)
                ordered = bits ^ cutlass.Uint32(0x80000000)
                if (bits & cutlass.Uint32(0x80000000)) != 0:
                    ordered = bits ^ cutlass.Uint32(0xFFFFFFFF)
                if ordered > prefix or (
                    ordered == prefix and index <= tie_index
                ):
                    (fused_route_scores_ptr + index).store(
                        cutlass.Float32.inf
                    )
                index += cutlass.Int32(128)
            self.cta_sync_barrier.arrive_and_wait()

        for group in range(0, num_groups, 1):
            group_start = group * 64
            if cutlass.const_expr(self.prefetch_summary):
                # AK/AV do not alias route scratch. Start the likely-needed
                # summary transfer before classification/compaction so their
                # scalar and warp work hides cp.async latency.
                self._load_summary_group(
                    gmem_copy, tAKgAK, tAVgAV, tKsK, tVsV,
                    tAKcAK, group, blocks, cutlass.Int32(0),
                )
            # Four warps classify four block centroids concurrently. Invalid
            # columns are marked exact so they are absent from the summary MMA.
            for item in cutlass.range_constexpr(16):
                column = warp_idx + item * 4
                block = group_start + column
                route_score = cutlass.Float32.zero
                if cutlass.const_expr(
                    not self.fused_topk_route and not self.external_route
                ):
                    if block < blocks:
                        for part in cutlass.range_constexpr(4):
                            dim = lane + part * 32
                            route_score += (
                                cutlass.Float32((qmean_ptr + dim).load())
                                * cutlass.Float32(mKC[batch, block, head, dim])
                            )
                        route_score = self._warp_reduce_sum(route_score)
                if lane == 0:
                    exact = block >= blocks
                    if block < blocks:
                        if cutlass.const_expr(self.fused_topk_route):
                            exact = block >= candidate_blocks
                            if block < candidate_blocks:
                                exact = (
                                    cutlass.Float32(
                                        (fused_route_scores_ptr + block).load()
                                    ) == cutlass.Float32.inf
                                )
                        elif cutlass.const_expr(self.external_route):
                            route_word = cutlass.Int32(
                                mRouteMask[
                                    batch,
                                    q_block,
                                    head,
                                    block // cutlass.Int32(32),
                                ]
                            )
                            route_bit = block % cutlass.Int32(32)
                            exact = (
                                ((route_word >> route_bit) & cutlass.Int32(1))
                                != cutlass.Int32(0)
                            )
                        else:
                            exact = route_score * scale_log2 > threshold
                        if block >= sink_start_block and block < sink_end_block:
                            exact = True
                        if cutlass.const_expr(self.local_block_radius == 0):
                            if block == q_block:
                                exact = True
                        elif cutlass.const_expr(self.local_block_radius > 0):
                            if (block >= q_block - self.local_block_radius
                                    and block <= q_block + self.local_block_radius):
                                exact = True
                    route_bias = -cutlass.Float32.inf
                    if block < blocks and not exact:
                        route_bias = (
                            cutlass.Float32(
                                mLM[batch, parent, head_local, block]
                            ) / softmax_scale
                        )
                    (route_ptr + column).store(route_bias)
            self.cta_sync_barrier.arrive_and_wait()
            if warp_idx == 0:
                preceding = cutlass.Int32.zero
                lane_mask_lt = cutlass.Int32(0x7FFFFFFF) >> (
                    cutlass.Int32(31) - lane
                )
                valid_count = cutlass.Int32.zero
                for word in cutlass.range_constexpr(2):
                    column = cutlass.Int32(word * 32) + lane
                    block = group_start + column
                    valid = block < blocks
                    exact = False
                    if valid:
                        exact = (
                            cutlass.Float32((route_ptr + column).load())
                            == -cutlass.Float32.inf
                        )
                    ballot = cutlass.Int32(cute.arch.vote_ballot_sync(exact))
                    rank = preceding + sol_attn_popc_b32(ballot & lane_mask_lt)
                    if exact:
                        (exact_indices_ptr + rank).store(block)
                    preceding += sol_attn_popc_b32(ballot)
                    if valid:
                        valid_count += 1
                valid_count = self._warp_reduce_sum(valid_count)
                if lane == 0:
                    exact_count_ptr.store(preceding)
                    skipped_ptr.store(valid_count - preceding)
            self.cta_sync_barrier.arrive_and_wait()

            if skipped_ptr.load() > 0:
                if cutlass.const_expr(not self.prefetch_summary):
                    self._load_summary_group(
                        gmem_copy, tAKgAK, tAVgAV, tKsK, tVsV,
                        tAKcAK, group, blocks, cutlass.Int32(0),
                    )
                cute.arch.cp_async_wait_group(0)
                self.cta_sync_barrier.arrive_and_wait()
                # Once the summary tile is resident, use its MMA/softmax work
                # to hide the first exact K/V transfer.
                if exact_count_ptr.load() > 0:
                    first_exact = exact_indices_ptr.load()
                    self._load_exact_block(
                        gmem_copy, tKgK, tVgV, tKsK, tVsV,
                        tKcK, first_exact, tokens, cutlass.Int32(1),
                    )
                self._compute_loaded_tile(
                    tiled_mma, copy_k, copy_v, tSsK, tSsV,
                    tSrQ, tSrK, tOrV, tSrK_view, tOrV_view,
                    acc_O, row_max, row_sum, score_coords,
                    route_ptr,
                    group_start, blocks, tokens, cutlass.Int32(-1),
                    scale_log2, cutlass.Int32(0), True,
                    group + 1 == num_groups and exact_count_ptr.load() == 0,
                )
            elif cutlass.const_expr(self.prefetch_summary):
                # Speculative all-exact groups must release stage zero before
                # its first physical K/V block overwrites the same storage.
                cute.arch.cp_async_wait_group(0)
                self.cta_sync_barrier.arrive_and_wait()

            # Stream selected physical blocks through the same accumulators.
            exact_count = exact_count_ptr.load()
            if exact_count > 0:
                stage = cutlass.Int32(1)
                if skipped_ptr.load() == 0:
                    stage = cutlass.Int32(0)
                    first_exact = exact_indices_ptr.load()
                    self._load_exact_block(
                        gmem_copy, tKgK, tVgV, tKsK, tVsV,
                        tKcK, first_exact, tokens, stage,
                    )
                for exact_index in range(0, exact_count, 1):
                    cute.arch.cp_async_wait_group(0)
                    next_stage = cutlass.Int32(1) - stage
                    self.cta_sync_barrier.arrive_and_wait()
                    if exact_index + 1 < exact_count:
                        next_block = (exact_indices_ptr + exact_index + 1).load()
                        self._load_exact_block(
                            gmem_copy, tKgK, tVgV, tKsK, tVsV,
                            tKcK, next_block, tokens, next_stage,
                        )
                    block = (exact_indices_ptr + exact_index).load()
                    self._compute_loaded_tile(
                        tiled_mma, copy_k, copy_v, tSsK, tSsV,
                        tSrQ, tSrK, tOrV, tSrK_view, tOrV_view,
                        acc_O, row_max, row_sum, score_coords,
                        route_ptr,
                        group_start, blocks, tokens, block,
                        scale_log2, stage, False,
                        group + 1 == num_groups
                        and exact_index + 1 == exact_count,
                    )
                    stage = next_stage

        self._store_output(
            mO, mLSE, batch, head, q_block, q_len, tidx,
            acc_O, row_max, row_sum, sQ, gmem_store, tiled_mma,
        )

    @cute.jit
    def _load_summary_group(
        self, gmem_copy, tAKgAK, tAVgAV, tKsK, tVsV,
        tAKcAK, group, blocks, stage,
    ):
        for row_part in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
            if cute.elem_less(tAKcAK[0, row_part, 0, group][1], blocks):
                cute.copy(gmem_copy, tAKgAK[None, row_part, None, group],
                          tKsK[None, row_part, None, stage])
                cute.copy(gmem_copy, tAVgAV[None, row_part, None, group],
                          tVsV[None, row_part, None, stage])
            else:
                tKsK[None, row_part, None, stage].fill(0)
                tVsV[None, row_part, None, stage].fill(0)
        cute.arch.cp_async_commit_group()

    @cute.jit
    def _load_exact_block(
        self, gmem_copy, tKgK, tVgV, tKsK, tVsV, tKcK, block, tokens, stage,
    ):
        for row_part in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
            if cute.elem_less(tKcK[0, row_part, 0, block][1], tokens):
                cute.copy(gmem_copy, tKgK[None, row_part, None, block],
                          tKsK[None, row_part, None, stage])
                cute.copy(gmem_copy, tVgV[None, row_part, None, block],
                          tVsV[None, row_part, None, stage])
            else:
                tKsK[None, row_part, None, stage].fill(0)
                tVsV[None, row_part, None, stage].fill(0)
        cute.arch.cp_async_commit_group()

    @cute.jit
    def _compute_loaded_tile(
        self, tiled_mma, copy_k, copy_v, tSsK, tSsV,
        tSrQ, tSrK, tOrV, tSrK_view, tOrV_view,
        acc_O, row_max, row_sum, score_coords, route,
        group_start, blocks, tokens, exact_block,
        scale_log2, stage, approximate: cutlass.Constexpr,
        final_tile,
    ):
        acc_S = cute.make_rmem_tensor(
            tiled_mma.get_slice(cute.arch.thread_idx()[0]).partition_shape_C((64, 64)),
            cutlass.Float32,
        )
        acc_S.fill(0.0)
        cute.copy(copy_k, tSsK[None, None, 0, stage], tSrK_view[None, None, 0])
        for kk in cutlass.range_constexpr(cute.size(tSsK.shape[2])):
            kk_next = (kk + 1) % cute.size(tSsK.shape[2])
            cute.copy(copy_k, tSsK[None, None, kk_next, stage],
                      tSrK_view[None, None, kk_next])
            cute.gemm(tiled_mma, acc_S, tSrQ[None, None, kk],
                      tSrK[None, None, kk], acc_S)

        scores = self._make_acc_tensor_mn_view(acc_S)
        coords = self._make_acc_tensor_mn_view(score_coords)
        out_mn = self._make_acc_tensor_mn_view(acc_O)
        for row in cutlass.range_constexpr(cute.size(row_max)):
            for col in cutlass.range_constexpr(cute.size(scores.shape[1])):
                local_col = coords[row, col][1]
                if cutlass.const_expr(approximate):
                    block = group_start + local_col
                    route_bias = cutlass.Float32(
                        (route + local_col).load()
                    )
                    if block >= blocks or route_bias == -cutlass.Float32.inf:
                        scores[row, col] = -cutlass.Float32.inf
                    else:
                        scores[row, col] += route_bias
                else:
                    token = exact_block * 64 + local_col
                    if token >= tokens:
                        scores[row, col] = -cutlass.Float32.inf

            score_row = scores[row, None].load()
            tile_max = score_row.reduce(
                cute.ReductionOp.MAX, -cutlass.Float32.inf, 0
            )
            tile_max = self._threadquad_reduce_max(tile_max)
            old_max = row_max[row]
            new_max = cute.arch.fmax(old_max, tile_max)
            alpha = cute.math.exp2((old_max - new_max) * scale_log2,
                                   fastmath=True)
            probabilities = cute.math.exp2(
                (score_row - new_max) * scale_log2, fastmath=True
            )
            tile_sum = probabilities.reduce(
                cute.ReductionOp.ADD, cutlass.Float32.zero, 0
            )
            row_sum[row] = row_sum[row] * alpha + tile_sum
            row_max[row] = new_max
            out_mn[row, None] = out_mn[row, None].load() * alpha
            scores[row, None] = probabilities

        rP = cute.make_fragment_like(acc_S, self._dtype)
        rP.store(acc_S.load().to(self._dtype))
        divided = cute.logical_divide(rP.layout, (None, None, 2))
        p_layout = cute.make_layout(
            ((divided.shape[0], divided.shape[2][0]), divided.shape[1],
             divided.shape[2][1]),
            stride=((divided.stride[0], divided.stride[2][0]),
                    divided.stride[1], divided.stride[2][1]),
        )
        tOrP = cute.make_tensor(rP.iterator, p_layout)
        cute.copy(copy_v, tSsV[None, None, 0, stage], tOrV_view[None, None, 0])
        for kk in cutlass.range_constexpr(cute.size(tOrP.shape[2])):
            kk_next = (kk + 1) % cute.size(tOrP.shape[2])
            cute.copy(copy_v, tSsV[None, None, kk_next, stage],
                      tOrV_view[None, None, kk_next])
            cute.gemm(tiled_mma, acc_O, tOrP[None, None, kk],
                      tOrV[None, None, kk], acc_O)
        if cutlass.const_expr(self.skip_final_tile_barrier):
            if not final_tile:
                self.cta_sync_barrier.arrive_and_wait()
        else:
            self.cta_sync_barrier.arrive_and_wait()

    @cute.jit
    def _store_output(
        self, mO, mLSE, batch, head, q_block, q_len, tidx,
        acc_O, row_max, row_sum, sQ, gmem_store, tiled_mma,
    ):
        out_mn = self._make_acc_tensor_mn_view(acc_O)
        for row in cutlass.range_constexpr(cute.size(row_sum)):
            total = self._threadquad_reduce_sum(row_sum[row])
            inv = cute.arch.rcp_approx(total)
            out_mn[row, None] = out_mn[row, None].load() * inv
            row_sum[row] = total

        rO = cute.make_fragment_like(acc_O, self._dtype)
        rO.store(acc_O.load().to(self._dtype))
        copy_o = cute.make_tiled_copy_C(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self._dtype),
            tiled_mma,
        )
        o_slice = copy_o.get_slice(tidx)
        tOrO = copy_o.retile(rO)
        tOsO = o_slice.partition_D(sQ)
        cute.copy(copy_o, tOrO, tOsO)
        self.cta_sync_barrier.arrive_and_wait()
        gO = cute.local_tile(
            mO[batch, None, head, None], (64, 128), (q_block, 0)
        )
        store_slice = gmem_store.get_slice(tidx)
        tOs = store_slice.partition_S(sQ)
        tOg = store_slice.partition_D(gO)
        cO = cute.local_tile(
            cute.make_identity_tensor(mO.layout.shape)[batch, None, head, None],
            (64, 128), (q_block, 0),
        )
        tOcO = store_slice.partition_D(cO)
        tOr = cute.make_fragment_like(tOg, self._dtype)
        cute.copy(gmem_store, tOs, tOr)
        for row_part in cutlass.range_constexpr(cute.size(tOg.shape[1])):
            if cute.elem_less(tOcO[0, row_part, 0][1], mO.shape[1]):
                cute.copy(gmem_store, tOr[None, row_part, None],
                          tOg[None, row_part, None])

        # LSE is not consumed by the fused virtual-query caller, but keeping
        # this epilogue produces better Ampere instruction scheduling than
        # eliding it and remains useful for diagnostics.
        coords = self._make_acc_tensor_mn_view(
            tiled_mma.get_slice(tidx).partition_C(
                cute.make_identity_tensor((64, 128))
            )
        )
        for row in cutlass.range_constexpr(cute.size(row_sum)):
            if coords[row, 0][1] == 0 and coords[row, 0][0] < q_len:
                mLSE[batch, q_block * 64 + coords[row, 0][0], head] = (
                    row_max[row] * 0.08838834764831845
                    + cute.math.log(row_sum[row], fastmath=True)
                )

    @cute.jit
    def _warp_reduce_sum(self, value: cutlass.Float32) -> cutlass.Float32:
        for offset in (16, 8, 4, 2, 1):
            value += cute.arch.shuffle_sync_down(
                value, offset=offset, mask=-1, mask_and_clamp=31
            )
        return value


def run(
    dtype: Type[cutlass.Numeric],
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    num_head: int,
    head_dim: int,
    softmax_scale: float = 1.0,
    m_block_size: int = 128,
    n_block_size: int = 128,
    num_threads: int = 128,
    is_causal: bool = False,
    warmup_iterations: int = 0,
    iterations: int = 1,
    skip_ref_check: bool = False,
    use_cold_l2: bool = False,
    **kwargs,
):
    import torch
    import cutlass.torch as cutlass_torch

    # Skip unsupported testcase
    if not FlashAttentionForwardAmpere.can_implement(
        dtype,
        head_dim,
        m_block_size,
        n_block_size,
        num_threads,
        is_causal,
    ):
        raise TypeError(
            f"Unsupported testcase {dtype}, {head_dim}, {m_block_size}, {n_block_size}, {num_threads}, {is_causal}"
        )

    print("Running Ampere SM80 FlashAttentionForward test with:")
    print(f"  dtype: {dtype}")
    print(f"  batch_size: {batch_size}")
    print(f"  seqlen_q: {seqlen_q}")
    print(f"  seqlen_k: {seqlen_k}")
    print(f"  num_head: {num_head}")
    print(f"  head_dim: {head_dim}")
    print(f"  softmax_scale: {softmax_scale}")
    print(f"  m_block_size: {m_block_size}")
    print(f"  n_block_size: {n_block_size}")
    print(f"  num_threads: {num_threads}")
    print(f"  is_causal: {is_causal}")
    print(f"  warmup_iterations: {warmup_iterations}")
    print(f"  iterations: {iterations}")
    print(f"  skip_ref_check: {skip_ref_check}")
    print(f"  use_cold_l2: {use_cold_l2}")

    # Create tensor Q/K/V/O
    def create_tensor(
        batch_size: int,
        seqlen: int,
        num_head: int,
        head_dim: int,
        dtype: Type[cutlass.Numeric],
    ) -> cute.Tensor:
        # (batch_size, seqlen, num_head, head_dim)
        shape = (batch_size, seqlen, num_head, head_dim)
        torch_tensor = (
            torch.empty(*shape, dtype=torch.int32)
            .random_(-2, 2)
            .to(dtype=cutlass_torch.dtype(dtype))
            .cuda()
        )
        # assume input is 16B aligned.
        cute_tensor = (
            from_dlpack(torch_tensor, assumed_align=16)
            .mark_layout_dynamic(leading_dim=3)
            .mark_compact_shape_dynamic(
                mode=3,
                stride_order=torch_tensor.dim_order(),
                divisibility=(128 // dtype.width),
            )
        )
        return cute_tensor, torch_tensor

    q, q_torch = create_tensor(batch_size, seqlen_q, num_head, head_dim, dtype)
    k, k_torch = create_tensor(batch_size, seqlen_k, num_head, head_dim, dtype)
    v, v_torch = create_tensor(batch_size, seqlen_k, num_head, head_dim, dtype)
    o, o_torch = create_tensor(batch_size, seqlen_q, num_head, head_dim, dtype)

    fa2_fwd = FlashAttentionForwardAmpere(
        head_dim,
        m_block_size,
        n_block_size,
        num_threads,
        is_causal,
    )

    # Get current CUDA stream from PyTorch
    torch_stream = torch.cuda.current_stream()
    # Get the raw stream pointer as a CUstream
    current_stream = cuda.CUstream(torch_stream.cuda_stream)
    # Pass compile options if needed
    compile_options = ""
    # compile the fa2 forward pass
    compiled_fa2_fwd = cute.compile(
        fa2_fwd, q, k, v, o, softmax_scale, current_stream, options=compile_options
    )

    if not skip_ref_check:
        compiled_fa2_fwd(q, k, v, o, softmax_scale, current_stream)
        torch.cuda.synchronize()
        q_ref = q_torch.permute(0, 2, 1, 3)
        k_ref = k_torch.permute(0, 2, 1, 3)
        v_ref = v_torch.permute(0, 2, 1, 3)
        torch.backends.cuda.enable_flash_sdp(enabled=True)
        ref_o = torch.nn.functional.scaled_dot_product_attention(
            q_ref, k_ref, v_ref, scale=softmax_scale, is_causal=is_causal
        ).permute(0, 2, 1, 3)
        torch.testing.assert_close(o_torch.cpu(), ref_o.cpu(), atol=1e-02, rtol=1e-04)
        print("Results verified successfully!")

    def generate_tensors():
        q_workspace, _ = create_tensor(batch_size, seqlen_q, num_head, head_dim, dtype)
        k_workspace, _ = create_tensor(batch_size, seqlen_k, num_head, head_dim, dtype)
        v_workspace, _ = create_tensor(batch_size, seqlen_k, num_head, head_dim, dtype)
        o_workspace, _ = create_tensor(batch_size, seqlen_q, num_head, head_dim, dtype)
        return testing.JitArguments(
            q_workspace,
            k_workspace,
            v_workspace,
            o_workspace,
            softmax_scale,
            current_stream,
        )

    workspace_count = 1
    if use_cold_l2:
        one_workspace_bytes = (
            q_torch.numel() * q_torch.element_size()
            + k_torch.numel() * k_torch.element_size()
            + v_torch.numel() * v_torch.element_size()
            + o_torch.numel() * o_torch.element_size()
        )
        workspace_count = testing.get_workspace_count(
            one_workspace_bytes, warmup_iterations, iterations
        )

    avg_time_us = testing.benchmark(
        compiled_fa2_fwd,
        workspace_generator=generate_tensors,
        workspace_count=workspace_count,
        stream=current_stream,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )

    return avg_time_us  # Return execution time in microseconds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="example of flash attention v2 with CuTe on GPU"
    )
    parser.add_argument("--dtype", type=cutlass.dtype, default=cutlass.BFloat16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seqlen_q", type=int, default=8192)
    parser.add_argument("--seqlen_k", type=int, default=8192)
    parser.add_argument("--num_head", type=int, default=16)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--softmax_scale", type=float, default=0.5)
    parser.add_argument("--m_block_size", type=int, default=128)
    parser.add_argument("--n_block_size", type=int, default=64)
    parser.add_argument("--num_threads", type=int, default=128)
    parser.add_argument("--is_causal", action="store_true", help="Enable causal mask")
    parser.add_argument("--warmup_iterations", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--skip_ref_check", action="store_true", help="Skip reference check"
    )
    parser.add_argument(
        "--use_cold_l2",
        action="store_true",
        default=False,
        help="Use circular buffer tensor sets to ensure L2 cold cache",
    )
    args = parser.parse_args()
    run(
        args.dtype,
        args.batch_size,
        args.seqlen_q,
        args.seqlen_k,
        args.num_head,
        args.head_dim,
        args.softmax_scale,
        args.m_block_size,
        args.n_block_size,
        args.num_threads,
        args.is_causal,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.use_cold_l2,
    )

    print("PASS")
