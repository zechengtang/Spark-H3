"""Opt-in QKV producer candidates for controlled ComfyUI A/B experiments.

Never imported by the production plugin. Core attention, plan, and reweight
remain untouched. Installation is confined to an experimental server process.
"""
import functools
import torch

def install(spark, *, chunk=4096, direct=True, views=True):
    import comfy.model_management
    import comfy.model_prefetch
    import comfy.quant_ops
    from comfy_kitchen.backends import cuda as ck
    make_layout=spark._make_spark_layout

    @functools.wraps(make_layout)
    def layout(*a,**kw):
        value=make_layout(*a,**kw)
        value.permutation._h3_video_span=spark._target_video_span(a[0],a[1])
        return value

    def producer(attn,x,rope_freqs,permutation):
        tokens=x.shape[0];heads=int(attn.heads);dim=int(attn.head_dim);inner=heads*dim
        q=torch.empty((1,tokens,heads,dim),device=x.device,dtype=x.dtype)
        k=torch.empty_like(q);v=torch.empty_like(q)
        qw=comfy.model_management.cast_to(attn.q_norm.weight,device=x.device)
        kw=comfy.model_management.cast_to(attn.k_norm.weight,device=x.device)
        rot=rope_freqs.shape[-3]*2
        span=getattr(permutation,'_h3_video_span',None)
        wrap=ck._wrap_for_dlpack
        stream=torch.cuda.current_stream(x.device).cuda_stream
        with comfy.model_prefetch.pause_malloc_graph():
            for start in range(0,tokens,chunk):
                stop=min(tokens,start+chunk);n=stop-start
                offset=None
                if views and span is not None:
                    lo,hi=span;nv=hi-lo
                    if stop<=nv:offset=lo
                    elif start>=nv and stop<=hi:offset=-nv
                    elif start>=hi:offset=0
                if offset is None:
                    indices=permutation[start:stop]
                    xc=x.index_select(0,indices);fc=rope_freqs.index_select(1,indices)
                else:
                    xc=x[start+offset:stop+offset];fc=rope_freqs[:,start+offset:stop+offset]
                projected=attn.qkv_proj(xc)
                qc,kc,vc=projected.split(inner,dim=-1)
                qc=qc.view(1,n,heads,dim);kc=kc.view(1,n,heads,dim)
                if direct:
                    ck._C.rms_rope(*(wrap(t) for t in (qc,kc,fc,qw,kw,q[:,start:stop],k[:,start:stop])),
                                   attn.q_norm.eps,stream,True,rot)
                else:
                    comfy.quant_ops.ck.rms_rope_split_half_(qc,kc,fc,qw,kw,epsilon=attn.q_norm.eps,rot_dim=rot)
                    q[:,start:stop].copy_(qc);k[:,start:stop].copy_(kc)
                v[:,start:stop].copy_(vc.view(1,n,heads,dim))
        return q,k,v

    spark._make_spark_layout=layout
    spark._native_spark_qkv=producer
