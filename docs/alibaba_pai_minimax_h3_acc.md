# Alibaba-PAI MiniMax-H3-Acc integration

Source: `alibaba-pai/MiniMax-H3-Acc-LoRAs`, pinned to Hugging Face revision
`335001fb9e5455d68a0caa18ec2e319072150328`.

| File | Target trunk | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `MiniMax-H3-FL2VA-Acc-8Step.safetensors` | FL2VA / T2VA | 1,372,450,680 | `0b29be7042d883970eb0c20774a9ba03d95669ed80a721bb4d21be8ea0d0a196` |
| `MiniMax-H3-Ref2VA-Acc-8Step.safetensors` | Ref2VA | 1,372,450,680 | `111c82e669f6e20e628228172edf39395f1a9fc3ad049793895e542c0f55b18c` |

Both releases are BF16 rank-64 trunk LoRAs with alpha 64 plus 32 PDD
video/audio output heads. They are not interchangeable: pair each file with
the matching base-model trunk.

## Diffusers

The official files are already in the released Diffusers PDD layout and do
not need weight conversion. Use `scripts/minimax_h3_pdd.py` to attach the
trunk adapters and parallel heads. The helper returns NFE 8; MiniMax-H3's
Diffusers scheduler includes the terminal sigma, so invoke the modular
pipeline with `num_inference_steps=nfe + 1`.

## ComfyUI

The installed `BSAI-ComfyUI-MiniMax-H3-PDD-Acc` node accepts the original
files directly. It converts trunk keys in memory, rebases the dense AdaLN
updates onto pruned-model curves, and installs the PDD head bank. Use
`MiniMaxH3PDDAccApply`, its `sigmas` output, plain Euler, CFG 1.0, and sigma
shifts 12/3. A stock LoRA loader drops essential PDD behavior and is invalid.

The Spark example is
`workflows/spark_h3_alibaba_pai_acc_fl2va_8step_lora_5p2s_t2va.json`.
