/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Independent split producer: Q/V tiles plus an already-normalized global K.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include "sol_layout.cuh"
namespace {
using namespace sol;
constexpr int HD=HEAD_DIM, BLK=BLOCK;
__device__ __forceinline__ void materialize_tile(const __nv_bfloat16*, __nv_bfloat16*, int,int,int,int) {}
__global__ void split_producer_kernel(
    const __nv_bfloat16* __restrict__ qkv, const __nv_bfloat16* raw_k, const __nv_bfloat16* raw_v,   // [M, 3*H*HD], chunk at token t0
    const float* __restrict__ fab,           // [T, rot, 2] packed rope coeffs
    const __nv_bfloat16* __restrict__ qw, const __nv_bfloat16* __restrict__ kw,
    __nv_bfloat16* __restrict__ qbf,
    __nv_bfloat16* __restrict__ kbf,
    __nv_bfloat16* __restrict__ vbf,
    const int32_t* __restrict__ qsample_map,
    __nv_bfloat16* __restrict__ qsample_out,
    const float* __restrict__ kmean,         // [H, HD] stale (may be zeros)
    const float* __restrict__ vscale,        // [H, HD] stale V scale (may be ~0 -> margin)
    int8_t* __restrict__ qiP, float* __restrict__ qs,
    int8_t* __restrict__ kiP, float2* __restrict__ ksb,
    int8_t* __restrict__ vTi, int8_t* __restrict__ vRow, __nv_bfloat16* __restrict__ vcT,
    float* __restrict__ ksumP,               // [H, NPAD, HD] block K sums (post-rope)
    int8_t* __restrict__ cen8, float* __restrict__ cens,
    float* __restrict__ qmean,               // [H, NPAD, HD] f32 block means (post-rope)
    float* __restrict__ vamax_next,          // [H, HD] atomicMax accumulator
    const int32_t* __restrict__ blen,        // [NTB] valid tokens per block, or null
    float rope_eps, int rot,
    int t0, int M, int T, int Tp, int H, int NPAD, int NQ, int fab_start, int k_head_tokens, int sq, int sk, int sv, int phase)
{
    __shared__ __align__(16) __nv_bfloat16 sT[BLK * LD_TILE];
    __shared__ __align__(16) float sred[HD];
    const int blk_local = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    const int tb0 = t0 + blk_local * BLK;              // absolute token start
    if (tb0 >= T) return;
    const int nblk = tb0 / BLK;                        // global 64-block index
    const int nrows = min(BLK, min(M - blk_local * BLK, T - tb0));   // rows that exist
    const int len = min(block_len_of(blen, nblk, T), nrows);         // rows that are live
    if (len <= 0) return;
    const int64_t row_stride = sq;
    const __nv_bfloat16* rows = qkv + (int64_t)(blk_local * BLK) * row_stride;
    const float* fab_t0 = fab + (int64_t)(tb0 - fab_start) * (rot * 2);

    // ---------------- Q phase ----------------
    if (phase & 1) {
    stage_tile64(sT, rows + (int64_t)h * HD, row_stride, len);
    __syncthreads();
    norm_rope_rows(sT, LD_TILE, len, fab_t0, qw, rope_eps, rot);
    __syncthreads();
    materialize_tile(sT, qbf, tb0, len, H, h);
    if (qsample_map) {
        for (int idx = tid; idx < BLK * (HD / 8); idx += HD) {
            const int t = idx / (HD / 8), c8 = (idx % (HD / 8)) * 8;
            if (t < len) {
                const int sample = qsample_map[tb0 + t];
                if (sample >= 0) {
                    const uint4 value = *reinterpret_cast<const uint4*>(sT + t * LD_TILE + c8);
                    *reinterpret_cast<uint4*>(qsample_out + ((size_t)sample * H + h) * HD + c8) = value;
                }
            }
        }
    }
    quant_q_rows(sT, len, nrows, qiP + ((size_t)tb0 * H + h) * HD, qs + (size_t)tb0 * H + h, H);
    __syncthreads();
    const size_t qrow = (size_t)h * NQ + nblk;
    const float c = centroid_quant(sT, len, sred, cen8 + qrow * HD, cens + qrow);
    qmean[((size_t)h * NPAD + nblk) * HD + tid] = c;
    __syncthreads();

    }
    // ---------------- K phase ----------------
    if (phase & 2) {
    stage_tile64(sT, raw_k + (int64_t)(blk_local * BLK) * sk + h * HD, sk, len);
    __syncthreads();
    // K has already undergone norm/RoPE exactly once.
    __syncthreads();
    if (k_head_tokens) {
        // Planner-only prefix in head-major order, ready for in-place projection.
        for (int idx = tid; idx < BLK * (HD / 8); idx += HD) {
            const int t = idx / (HD / 8), c8 = (idx % (HD / 8)) * 8;
            if (t < len && tb0 + t < k_head_tokens) {
                const uint4 value = *reinterpret_cast<const uint4*>(sT + t * LD_TILE + c8);
                *reinterpret_cast<uint4*>(kbf + ((size_t)h * k_head_tokens + tb0 + t) * HD + c8) = value;
            }
        }
    } else {
        materialize_tile(sT, kbf, tb0, len, H, h);
    }
    {
        // block K sums (post-rope, uncentered) for the pooled tensors
        float sk = 0.f;
        for (int t = 0; t < len; ++t) sk += __bfloat162float(sT[t * LD_TILE + tid]);
        ksumP[((size_t)h * NPAD + nblk) * HD + tid] = sk;
    }
    const size_t dst0 = (size_t)h * Tp + nblk * BLK;
    quant_k_rows(sT, len, kmean + (size_t)h * HD, nullptr, kiP + dst0 * HD, ksb + dst0);
    __syncthreads();

    }
    // ---------------- V phase ----------------
    if (phase & 4) {
    stage_tile64(sT, raw_v + (int64_t)(blk_local * BLK) * sv + h * HD, sv, len);
    __syncthreads();
    materialize_tile(sT, vbf, tb0, len, H, h);
    {
        const int d = tid;
        const float inv = 1.f / vscale[(size_t)h * HD + d];
        float sv = 0.f, av = 0.f;
        // vTi: raw channel rows, perm_d on the KEY axis per 64-block
        __align__(16) int8_t col[BLK];
        for (int t = 0; t < BLK; ++t) {
            const float x = (t < len) ? __bfloat162float(sT[t * LD_TILE + d]) : 0.f;
            sv += x; av = fmaxf(av, fabsf(x));
            col[perm_d(t)] = q8(x, inv);
            if (vRow) vRow[((size_t)h * Tp + nblk * BLK + t) * HD + d] = col[perm_d(t)];   // row-major copy (token routing)
        }
        const size_t vbase = ((size_t)h * HD + d) * Tp + nblk * BLK;
        #pragma unroll
        for (int c = 0; c < BLK; c += 16)
            *reinterpret_cast<uint4*>(vTi + vbase + c) = *reinterpret_cast<const uint4*>(col + c);
        vcT[((size_t)h * HD + d) * NPAD + nblk] = __float2bfloat16(sv);
        atomicMax(reinterpret_cast<unsigned int*>(&vamax_next[(size_t)h * HD + d]),
                  __float_as_uint(av));
    }
}
}
}
extern "C" int spark_split_produce(
    const void* q, const void* k, const void* v, const void* fab, const void* qw,
    const void* km, const void* vs, void* qi, void* qs, void* ki, void* ks,
    void* vt, void* vr, void* vc, void* ksum, void* cen, void* cens,
    void* qm, void* vmax, float eps, int rot, int t0, int M, int T, int Tp,
    int H, int NPAD, int NQ, int sq, int sk, int sv, int phase, cudaStream_t stream) {
    split_producer_kernel<<<dim3((M+63)/64,H),128,0,stream>>>(
        (const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const __nv_bfloat16*)v,
        (const float*)fab,(const __nv_bfloat16*)qw,nullptr,
        nullptr,nullptr,nullptr,nullptr,nullptr,(const float*)km,(const float*)vs,
        (int8_t*)qi,(float*)qs,(int8_t*)ki,(float2*)ks,(int8_t*)vt,(int8_t*)vr,
        (__nv_bfloat16*)vc,(float*)ksum,(int8_t*)cen,(float*)cens,(float*)qm,
        (float*)vmax,nullptr,eps,rot,t0,M,T,Tp,H,NPAD,NQ,0,0,sq,sk,sv,phase);
    return (int)cudaGetLastError();
}
namespace {
__global__ void normalize_kernel(const __nv_bfloat16* raw, const float* fab,
    const __nv_bfloat16* weight, __nv_bfloat16* out,
    int M, int H, int stride, int rot, float eps) {
    __shared__ __align__(16) __nv_bfloat16 tile[64*sol::LD_TILE];
    int b=blockIdx.x,h=blockIdx.y,len=min(64,M-b*64);
    sol::stage_tile64(tile,raw+(int64_t)b*64*stride+h*128,stride,len);
    __syncthreads();
    sol::norm_rope_rows(tile,sol::LD_TILE,len,fab+(int64_t)b*64*rot*2,weight,eps,rot);
    __syncthreads();
    for(int t=0;t<len;++t) out[((int64_t)(b*64+t)*H+h)*128+threadIdx.x]=tile[t*sol::LD_TILE+threadIdx.x];
}
}
extern "C" int spark_split_normalize(const void* raw,const void* fab,const void* weight,
    void* out,int M,int H,int stride,int rot,float eps,cudaStream_t stream) {
    normalize_kernel<<<dim3((M+63)/64,H),128,0,stream>>>((const __nv_bfloat16*)raw,
        (const float*)fab,(const __nv_bfloat16*)weight,(__nv_bfloat16*)out,M,H,stride,rot,eps);
    return (int)cudaGetLastError();
}
