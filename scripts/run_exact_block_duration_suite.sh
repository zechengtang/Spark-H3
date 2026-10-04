#!/usr/bin/env bash
set -euo pipefail

repo=/mnt/CFS/tangzecheng/repos/Spark-H3
root10=/mnt/CFS/tangzecheng/experiments/exact_block_radius_qk_10s768p_25p_20261004
root14=/mnt/CFS/tangzecheng/experiments/spark_radius_q_from_k_14p4s768p_25p_20261004
current_pid=${1:?current 10s generation PID is required}
status=/mnt/CFS/tangzecheng/experiments/exact_block_duration_suite_20261004.status

cd "$repo"
trap 'printf "failed %s\n" "$(date -u +%FT%TZ)" > "$status"' ERR
printf "waiting_10s_generation %s\n" "$(date -u +%FT%TZ)" > "$status"
while kill -0 "$current_pid" 2>/dev/null; do sleep 30; done

test "$(python -c "import json; print(json.load(open('$root10/status.json'))['status'])")" = complete
printf "running_10s_quality %s\n" "$(date -u +%FT%TZ)" > "$status"
EXACT_QUALITY_FRAMES=240 python scripts/exact_block_radius_quality.py run \
  --experiment "$root10" \
  --arms q_reuses_k_exact_radius0 k_reuses_q_exact_radius0

printf "running_10s_vbench %s\n" "$(date -u +%FT%TZ)" > "$status"
python scripts/exact_block_radius_vbench.py all \
  --experiment "$root10" \
  --methods dense q_reuses_k_exact_radius0 k_reuses_q_exact_radius0 \
  --gpus 0 1 2 3 4 5 6 7

printf "running_14p4s_generation %s\n" "$(date -u +%FT%TZ)" > "$status"
python scripts/exact_block_radius_14p4s768p_25p.py run \
  --arms dense baseline q_reuses_k_layout q_reuses_k_exact_radius0

test "$(python -c "import json; print(json.load(open('$root14/status.json'))['status'])")" = complete
printf "running_14p4s_quality %s\n" "$(date -u +%FT%TZ)" > "$status"
EXACT_QUALITY_FRAMES=360 python scripts/exact_block_radius_quality.py run \
  --experiment "$root14" \
  --arms baseline q_reuses_k_layout q_reuses_k_exact_radius0

printf "running_14p4s_vbench %s\n" "$(date -u +%FT%TZ)" > "$status"
python scripts/exact_block_radius_vbench.py all \
  --experiment "$root14" \
  --methods dense baseline q_reuses_k_layout q_reuses_k_exact_radius0 \
  --gpus 0 1 2 3 4 5 6 7

printf "complete %s\n" "$(date -u +%FT%TZ)" > "$status"
