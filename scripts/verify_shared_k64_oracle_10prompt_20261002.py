#!/usr/bin/env python3
"""Online, read-only Dense-trajectory routing audit on two GPUs.

Attention processors inspect post-RoPE Q/K and then invoke the unchanged Dense
processor. No sparse output is fed back into the trajectory. Each Q64 sample
has an exact fixed-layout K64 mass oracle at precisely the native 10% budget.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
NAME = "shared_k64_oracle_10prompt_allsteps_alllayers_20261002_v2"
ROOT = Path("/autodl-fs/data/h3_experiments") / NAME
CASES = (1, 6, 11, 15, 21, 27, 32, 38, 44, 50)
EVALUATIONS = tuple(range(19))
LAYERS = tuple(range(50))
HEADS = (0, 1, 2, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 49, 55)
QBLOCKS = 16
SEED = 42
PYTHON = "/root/miniconda3/bin/python"
BENCH = REPO.parent / "MiniMax-H3-Benchmark"
SAMPLES = BENCH / "vbench_core5_percent_subsets/20pct/samples.json"
CACHE = Path("/autodl-fs/data/h3_outputs/vbench20pct_768p10s_seed42_20260913")


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    if ROOT.exists():
        raise FileExistsError(ROOT)
    ROOT.mkdir(parents=True)
    for gpu in range(2):
        out = ROOT / f"workflow_{gpu}"
        out.mkdir()
        for name in ("conditioning_cache", "conditioning_manifest.json"):
            (out / name).symlink_to(CACHE / name)
    records = json.loads(SAMPLES.read_text())
    write(ROOT / "protocol.json", {
        "status": "running", "started_unix": time.time(),
        "cases": [r for r in records if r["index"] in CASES],
        "seed": SEED, "requested_steps": 20, "actual_evaluations": 19,
        "requested_frames": 120, "resolution": [1344, 768],
        "evaluations": EVALUATIONS, "layers": LAYERS, "heads": HEADS,
        "model_total_heads": 56,
        "query_blocks_per_head": QBLOCKS, "expected_rows": 10 * 19 * 50 * len(HEADS) * QBLOCKS,
        "actual_sparse_rows": 10 * 15 * 49 * len(HEADS) * QBLOCKS,
        "trajectory": "unchanged native compiled Diffusers Dense; no sparse feedback",
        "layout": "current fanout16 flat midpoint32 reblock, Q64/K64",
        "budget": "round(0.10 * complete_video_K64_blocks)",
        "primary_normalization": "complete video keys only, matched to earlier 96.13% diagnostic",
        "secondary_normalization": "all video/context/tail keys; oracle reranked with row denominators",
        "scope": "5s only, one seed per prompt; no end-to-end quality/speed inference",
        "git_revision": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(p): sha(p) for p in (
            Path(__file__), REPO / "h3_sparse_attention/spark_integration.py",
            REPO / "h3_sparse_attention/processor.py", REPO / "h3_sparse_attention/landmark_tree_v2.py")},
    })
    shutil.copy2(__file__, ROOT / "runner_source.py")


class CurrentRoute:
    """Call the current production reblock builder, with its reusable plan."""
    def __init__(self):
        from h3_sparse_attention import H3SparseAttentionConfig
        from h3_sparse_attention.processor import _Controller
        self.cfg = H3SparseAttentionConfig.spark(
            20, warmup_percent=20, sol_dense_layers=1,
            sol_force_local_blocks=False, sol_log_density=False)
        self.controller = _Controller(self.cfg)
        self.validated = False

    def config(self):
        return self.cfg

    def production_partition(self, q, k, grid, cfg):
        import torch
        from types import SimpleNamespace
        from h3_sparse_attention.spark_integration import _landmark_tree_v2_qk_block_permutations
        qb, kb = [v.permute(1, 0, 2)[None].contiguous() for v in (q, k)]
        result = _landmark_tree_v2_qk_block_permutations(
            self.controller, qb, kb,
            SimpleNamespace(grid=tuple(grid), video_tokens=q.shape[1]))
        qp, kp = result[0][0].clone(), result[2][0].clone()
        if not self.validated:
            identity = torch.arange(q.shape[1], device=q.device).expand_as(qp)
            for p, inv in ((qp, result[1][0]), (kp, result[3][0])):
                assert torch.equal(p.sort(-1).values, identity)
                assert torch.equal(inv.gather(-1, p), identity)
            self.validated = True
        return qp, kp, {"query": result[4], "key": result[5]}


class Collector:
    def __init__(self, transformer, case, base):
        self.transformer, self.case, self.base = transformer, case, base
        self.evaluation = -1
        self.video_rows = None
        self.originals = []
        self.cfg = base.config()
        self.units = 0

    def begin(self, _module, args, kwargs):
        import torch
        self.evaluation += 1
        if self.video_rows is not None:
            return
        tags = kwargs.get("token_tags", args[5] if len(args) > 5 else None)
        positions = kwargs.get("position_ids", args[6] if len(args) > 6 else None)
        assert tags is not None and positions is not None
        rows = torch.nonzero(tags == 0, as_tuple=False).flatten()
        pos = positions.index_select(0, rows)
        self.grid = tuple(int(torch.unique(pos[:, axis]).numel()) for axis in range(3))
        self.video_rows = rows
        complete = rows.numel() // 64
        selected = torch.zeros(tags.numel(), device=tags.device, dtype=torch.bool)
        selected[rows[:complete * 64]] = True
        self.other_rows = (~selected).nonzero(as_tuple=False).flatten()

    def install(self):
        self.hook = self.transformer.register_forward_pre_hook(self.begin, with_kwargs=True)
        collector = self
        for layer in LAYERS:
            attn = self.transformer.transformer_blocks[layer].attn
            original = attn.get_processor()
            self.originals.append((attn, original))

            class Processor:
                def __init__(self, layer, original):
                    self.layer, self.original = layer, original

                def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
                    collector.record(self.layer, attn, hidden_states, rotary_emb)
                    return self.original(attn, hidden_states, rotary_emb, attention_mask)

            attn.set_processor(Processor(layer, original))

    def remove(self):
        self.hook.remove()
        for attn, original in self.originals:
            attn.set_processor(original)

    def record(self, layer, attn, hidden, rotary):
        import torch
        from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb
        start = time.perf_counter()
        if attn.fused_projections:
            rawq, rawk, _ = attn.to_qkv(hidden).chunk(3, dim=-1)
        else:
            rawq, rawk = attn.to_q(hidden), attn.to_k(hidden)
        qfull = attn.norm_q(rawq.unflatten(-1, (attn.heads, -1)))
        kfull = attn.norm_k(rawk.unflatten(-1, (attn.heads, -1)))
        if rotary is not None:
            qfull = _apply_rotary_emb(qfull, *rotary)
            kfull = _apply_rotary_emb(kfull, *rotary)
        head_ids = torch.tensor(HEADS, device=hidden.device)
        q = qfull[0].index_select(1, head_ids).index_select(0, self.video_rows).permute(1, 0, 2).contiguous()
        k = kfull[0].index_select(1, head_ids).index_select(0, self.video_rows).permute(1, 0, 2).contiguous()
        qp, kp, metadata = self.base.production_partition(q, k, self.grid, self.cfg)
        q = q.gather(1, qp[..., None].expand_as(q)).float()
        k = k.gather(1, kp[..., None].expand_as(k)).float()
        heads, tokens, dim = q.shape
        complete, cut = tokens // 64, tokens // 64 * 64
        quota = max(1, round(0.1 * complete))
        ids = torch.linspace(0, complete - 1, QBLOCKS, device=q.device).round().long().unique()
        assert heads == len(HEADS) and len(ids) == QBLOCKS
        qblocks = q[:, :cut].reshape(heads, complete, 64, dim)
        kblocks = k[:, :cut].reshape(heads, complete, 64, dim)
        # Reblocking can move tokens into/out of its incomplete final block.
        # Use those exact permuted tail keys, rather than the original video tail.
        context_rows = self.other_rows[~torch.isin(self.other_rows, self.video_rows)]
        context = kfull[0].index_select(1, head_ids).index_select(0, context_rows).permute(1, 0, 2).float()
        others = torch.cat((k[:, cut:], context), dim=1)
        rows = []
        scale = dim ** -0.5
        for head in range(heads):
            query = qblocks[head, ids]
            logits = query.reshape(-1, dim) @ k[head, :cut].T * scale
            p = logits.softmax(-1)
            block_mass = p.reshape(QBLOCKS, 64, complete, 64).sum(-1).mean(1)
            mean_scores = query.mean(1) @ kblocks[head].mean(1).T * scale
            native_ids = mean_scores.topk(quota, dim=-1).indices
            oracle_ids = block_mass.topk(quota, dim=-1).indices
            native = block_mass.gather(-1, native_ids).sum(-1)
            oracle = block_mass.gather(-1, oracle_ids).sum(-1)
            logz = torch.logsumexp(logits, -1)
            if others.shape[1]:
                other_logits = query.reshape(-1, dim) @ others[head].T * scale
                all_logz = torch.logaddexp(logz, torch.logsumexp(other_logits, -1))
                weight = (logz - all_logz).exp().reshape(QBLOCKS, 64)
            else:
                weight = torch.ones((QBLOCKS, 64), device=q.device)
            full_mass = (p.reshape(QBLOCKS, 64, complete, 64).sum(-1) * weight[..., None]).mean(1)
            full_ids = full_mass.topk(quota, dim=-1).indices
            full_native = full_mass.gather(-1, native_ids).sum(-1)
            full_oracle = full_mass.gather(-1, full_ids).sum(-1)
            recall = (native_ids[:, :, None] == oracle_ids[:, None, :]).any(-1).float().mean(-1)
            values = torch.stack((native, oracle, native / oracle, oracle - native,
                                  recall, full_native, full_oracle, full_native / full_oracle,
                                  full_oracle - full_native, weight.mean(-1)), -1).cpu().tolist()
            names = ("native_mass", "oracle_mass", "efficiency", "mass_gap", "block_recall",
                     "full_native_mass", "full_oracle_mass", "full_efficiency", "full_mass_gap", "video_fraction")
            for qid, numbers in zip(ids.cpu().tolist(), values):
                rows.append({"head": HEADS[head], "query_block": qid, **dict(zip(names, numbers))})
        elapsed = time.perf_counter() - start
        target = ROOT / "units" / f"case{self.case:02d}_eval{self.evaluation:02d}_layer{layer:02d}.json"
        write(target, {"case": self.case, "evaluation": self.evaluation, "layer": layer,
                       "sparse_scope": self.evaluation >= 4 and layer >= 1,
                       "grid": self.grid, "video_tokens": tokens, "complete_blocks": complete,
                       "quota": quota, "context_tokens": context.shape[1], "tail_tokens": tokens-cut,
                       "diagnostic_seconds": elapsed, "rows": rows})
        self.units += 1
        if layer == 49:
            print("EVALUATION", self.case, self.evaluation, self.units, flush=True)


def worker(gpu):
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(BENCH / "scripts"))
    import _impl_bootstrap
    import minimax_h3_vbench_4gpu_pipeline as pipeline
    # Construct/validate configuration before the expensive model load.
    base = CurrentRoute()
    import dataclasses
    write(ROOT / f"route_config_{gpu}.json", dataclasses.asdict(base.config()))
    assigned = CASES[gpu::2]
    args = pipeline.build_parser().parse_args([
        "denoise", "--samples", str(SAMPLES), "--output", str(ROOT / f"workflow_{gpu}"),
        "--method", "dense", "--case-indices", *map(str, assigned), "--steps", "20",
        "--frames", "120", "--height", "768", "--width", "1344", "--workers", "1"])
    cases = pipeline.load_cases(SAMPLES, list(assigned), expected_indices=tuple(range(1, 51)))
    workflow, states = pipeline.configure_denoise_workflow(args, cases)
    pipe, manager, acceleration, placement = pipeline.load_denoiser(args, workflow)
    write(ROOT / f"worker_{gpu}.json", {"status": "running", "cases": assigned,
          "device": torch.cuda.get_device_name(), "placement": placement,
          "compiled_blocks": len(acceleration._forward_originals)})
    try:
        for case, state in zip(cases, states):
            index = case["index"]
            done = ROOT / "cases" / f"case{index:02d}.json"
            if done.exists():
                continue
            # A partially completed case is rerun from its Dense initial state.
            collector = Collector(pipe.transformer, index, base)
            collector.install()
            state = pipeline.clone_state(state)
            state.values["prompt_embeds"] = state.values["prompt_embeds"].to("cuda")
            begin = time.perf_counter()
            try:
                with torch.inference_mode():
                    output = pipe(state=state, num_frames=120, height=768, width=1344,
                                  num_inference_steps=20,
                                  generator=torch.Generator(device="cpu").manual_seed(SEED),
                                  output=["latents", "audio_latents"])
                assert collector.evaluation == 18 and collector.units == 950
                assert all(torch.isfinite(output[n]).all().item() for n in ("latents", "audio_latents"))
                write(done, {"status": "complete", "case": case, "seconds": time.perf_counter()-begin,
                             "units": collector.units, "finite": True,
                             "conditioning_shape": list(state.values["prompt_embeds"].shape)})
                del output
                print("CASE_COMPLETE", index, flush=True)
            finally:
                collector.remove()
            del state, collector
        write(ROOT / f"worker_{gpu}.json", {"status": "complete", "cases": assigned})
    finally:
        acceleration.remove()


METRICS = ("native_mass", "oracle_mass", "efficiency", "mass_gap", "block_recall",
           "full_native_mass", "full_oracle_mass", "full_efficiency", "full_mass_gap", "video_fraction")


def summarize(rows):
    import numpy as np
    matrix = np.array([[r[n] for n in METRICS] for r in rows])
    result = {n: float(matrix[:, i].mean()) for i, n in enumerate(METRICS)}
    result["rows"] = len(rows)
    result["efficiency_quantiles"] = {str(p): float(np.percentile(matrix[:, 2], p)) for p in (0, 1, 5, 10, 50, 90, 95, 99, 100)}
    result["fraction_efficiency_below"] = {str(v): float((matrix[:, 2] < v).mean()) for v in (.8, .9, .95)}
    return result


def aggregate():
    import numpy as np
    for case in CASES:
        assert (ROOT / "cases" / f"case{case:02d}.json").exists(), case
    rows = []
    units = []
    for path in sorted((ROOT / "units").glob("*.json")):
        unit = json.loads(path.read_text())
        units.append(unit)
        for row in unit["rows"]:
            rows.append({"case": unit["case"], "evaluation": unit["evaluation"],
                         "layer": unit["layer"], "sparse_scope": unit["sparse_scope"], **row})
    assert len(units) == 9500 and len(rows) == 10 * 19 * 50 * len(HEADS) * QBLOCKS
    sparse = [r for r in rows if r["sparse_scope"]]
    assert len(sparse) == 10 * 15 * 49 * len(HEADS) * QBLOCKS
    groups = {}
    for dimension in ("case", "evaluation", "layer", "head"):
        groups[dimension] = {str(key): summarize([r for r in sparse if r[dimension] == key])
                             for key in sorted({r[dimension] for r in sparse})}
    # Joint strata reveal weak layers at particular denoising stages that a
    # marginal layer or evaluation average could hide.
    joint_rows = {}
    for row in sparse:
        key = f"eval{row['evaluation']:02d}_layer{row['layer']:02d}"
        joint_rows.setdefault(key, []).append(row)
    joint = {key: summarize(value) for key, value in joint_rows.items()}
    per_case = np.array([v["efficiency"] for v in groups["case"].values()])
    boot = np.random.default_rng(42).choice(per_case, size=(10000, len(per_case)), replace=True).mean(1)
    result = {"status": "complete", "all": summarize(rows), "sparse": summarize(sparse),
              "dense_only": summarize([r for r in rows if not r["sparse_scope"]]),
              "sparse_prior_heads_0_to_3": summarize([r for r in sparse if r["head"] < 4]),
              "sparse_added_heads": summarize([r for r in sparse if r["head"] >= 4]),
              "restricted_prior_step_layer_head_scope": summarize([
                  r for r in sparse if r["head"] < 4 and r["evaluation"] in (4, 11, 18)
                  and r["layer"] in (12, 24, 36, 48)]),
              "sparse_groups": groups,
              "sparse_evaluation_layer_groups": joint,
              "prompt_cluster_bootstrap_efficiency_95ci": np.percentile(boot, [2.5, 97.5]).tolist(),
              "prior_2prompt_efficiency": .9612575,
              "sample_semantics": "equal Q64/head units; ratio averaged per unit; not independent prompt replicates",
              "geometry": [{n: units[0][n] for n in ("grid", "video_tokens", "complete_blocks", "quota")}],
              "case_records": [json.loads((ROOT / "cases" / f"case{i:02d}.json").read_text()) for i in CASES]}
    write(ROOT / "results.json", result)
    s = result["sparse"]
    lines = ["# Shared Q64/K64 oracle, expanded 10-prompt diagnostic", "",
             "Dense trajectory, seed42, 5s 1344x768, all19 evaluations/all50 layers/16 of 56 heads/16 Q64 blocks.",
             "Primary conclusion uses eval4–18 and layers1–49 (1,881,600 units).", "",
             f"Native mass {s['native_mass']:.6f}; oracle mass {s['oracle_mass']:.6f}; "
             f"mean efficiency {s['efficiency']:.6f}; absolute mass gap {s['mass_gap']:.6f}.",
             f"All-key-normalized efficiency {s['full_efficiency']:.6f}; mass gap {s['full_mass_gap']:.6f}.",
             f"Prompt-cluster bootstrap CI: {result['prompt_cluster_bootstrap_efficiency_95ci']}", "",
             "| Case | Native mass | Oracle mass | Efficiency | Gap |", "|---|---:|---:|---:|---:|"]
    for key, value in groups["case"].items():
        lines.append(f"| {key} | {value['native_mass']:.4f} | {value['oracle_mass']:.4f} | {value['efficiency']:.4f} | {value['mass_gap']:.4f} |")
    lines += ["", "Full step/layer/head groups and quantiles are in results.json.",
              "Mass oracles optimize retained probability, not output-vector error or generated quality.",
              "Ten prompts, one seed, and one length do not establish a universal bound."]
    (ROOT / "REPORT.md").write_text("\n".join(lines)+"\n")
    protocol = json.loads((ROOT / "protocol.json").read_text())
    protocol.update(status="complete", completed_unix=time.time(),
                    analysis_source_sha256=sha(Path(__file__)),
                    model="/autodl-fs/data/models/MiniMax-H3", model_dtype="bfloat16")
    shutil.copy2(__file__, ROOT / "analysis_source.py")
    write(ROOT / "protocol.json", protocol)
    print(json.dumps({"sparse": s, "confidence_interval": result["prompt_cluster_bootstrap_efficiency_95ci"]}), flush=True)


def run():
    if not ROOT.exists():
        prepare()
    write(ROOT / "effective_source.json", {
        "started_unix": time.time(), "runner_sha256": sha(Path(__file__)),
        "reblock_builder_sha256": sha(REPO / "h3_sparse_attention/spark_integration.py"),
        "note": "Current production Spark config/builder; initial startup used obsolete frozen-config keywords and produced no units."})
    shutil.copy2(__file__, ROOT / "effective_runner_source.py")
    (ROOT / "logs").mkdir(exist_ok=True)
    def launch(gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), H3_IMPL_REPO=str(REPO),
                   OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")
        with (ROOT / "logs" / f"worker_{gpu}.log").open("a") as log:
            p = subprocess.Popen([PYTHON, str(Path(__file__).resolve()), "worker", str(gpu)],
                                 env=env, stdout=log, stderr=subprocess.STDOUT)
            write(ROOT / f"process_{gpu}.json", {"pid": p.pid, "gpu": gpu})
            return p.wait()
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        codes = list(pool.map(launch, range(2)))
    if any(codes):
        raise RuntimeError(f"worker return codes: {codes}")
    aggregate()


def progress():
    paths = sorted((ROOT / "units").glob("*.json"))
    cases = sorted((ROOT / "cases").glob("*.json"))
    latest = {}
    for path in paths:
        latest[path.name.split("_", 1)[0]] = path
    print(json.dumps({"units": len(paths), "expected": 9500,
                      "completed_cases": len(cases), "latest": {
        key: {"file": path.name, "efficiency": statistics.mean(
            r["efficiency"] for r in json.loads(path.read_text())["rows"])}
        for key, path in latest.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "worker":
        worker(int(sys.argv[2]))
    else:
        globals()[command]()
