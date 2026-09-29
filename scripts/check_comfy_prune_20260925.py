"""Compare pruning modes against outputs saved by the original CUDA binary."""
import argparse
import json
from pathlib import Path

import torch
from comfy_kitchen.backends import cuda as ck


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    results = {}
    for dtype in (torch.bfloat16, torch.float16):
        for tokens, video in ((257, 193), (4097, 4000)):
            torch.manual_seed(571)
            q, k, v = [torch.randn(1, tokens, 2, 128, device="cuda", dtype=dtype)
                       for _ in range(3)]
            blocks = (tokens + 63) // 64
            cases = {
                "topk_tail_sink": dict(topk_ratio=0.1, sink_blocks=[video // 64, blocks],
                                       sink_q=[video // 64, blocks]),
                "topk_interior_sink": dict(topk_ratio=0.1, sink_blocks=[1, 3], sink_q=[0, 1]),
                "tau": dict(topk_ratio=0.0, tau=1.0, sink_blocks=[1, 3]),
                "all_exact": dict(topk_ratio=1.0),
                "all_summary": dict(topk_ratio=0.0, tau=1e9),
                "all_sink": dict(topk_ratio=0.1, sink_blocks=[0, blocks], sink_q=[0, blocks]),
            }
            perm = torch.stack([torch.randperm(video, device="cuda") for _ in range(2)])[None].int()
            for name, kw in cases.items():
                for reblock in (False, True):
                    opts = dict(kw)
                    if reblock:
                        opts.update(query_permutation=perm, key_permutation=perm)
                    key = f"{dtype}/{tokens}/{name}/reblock={reblock}"
                    results[key] = ck.spark_attn(q, k, v, video_tokens=video, **opts).cpu()
    if args.save:
        torch.save(results, args.reference)
        print(json.dumps({"saved": str(args.reference), "cases": len(results)}))
    else:
        reference = torch.load(args.reference, weights_only=True)
        failures = {k: float((v.float() - reference[k].float()).abs().max())
                    for k, v in results.items() if not torch.equal(v, reference[k])}
        print(json.dumps({"cases": len(results), "bitwise_failures": failures}))
        if failures:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
