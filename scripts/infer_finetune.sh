#!/usr/bin/env bash
# Fine-tuned stitch and HM scores on one GPU.
# nuScenes is split across several processes. Lyft and WideDrive3 run beside it,
# and a finished dataset is scored while the others are still stitching.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="/home/test/anaconda3/envs/rewarp/bin/python"
GPU=0
OUT="$ROOT/output/finetune"
SPLIT="$ROOT/logs/ft/splits"
LOG="$ROOT/logs/ft"
WEIGHT="$ROOT/pretrained/fine-tuning"
DATA="/home/test/LIVA/XZP/ImageStitch/RopStitch_infer/datasets"
mkdir -p "$SPLIT" "$LOG" "$OUT"

python3 - "$DATA/nuscenes/processed_10Hz_v2/nuScenes_Val.txt" "$SPLIT" << 'PY'
import sys
from pathlib import Path
src, out = Path(sys.argv[1]), Path(sys.argv[2])
rows = [line.strip() for line in src.read_text().splitlines() if line.strip()]
parts = 4
for index in range(parts):
    chunk = rows[index::parts]
    (out / f"nuscenes_{index}.txt").write_text("\n".join(chunk) + "\n")
    print(f"nuscenes_{index}: {len(chunk)}")
PY

run() {
  local name="$1"
  shift
  echo "=== ${name} start $(date '+%F %T') ===" >> "$LOG/${name}.log"
  "$@" >> "$LOG/${name}.log" 2>&1 || echo "FAIL ${name}" >> "$LOG/${name}.log"
  echo "=== ${name} done $(date '+%F %T') ===" >> "$LOG/${name}.log"
}

echo "=== finetune infer start $(date '+%F %T') gpu ${GPU} ===" | tee -a "$LOG/driver.log"
pids=()
for index in 0 1 2 3; do
  run "nuscenes_${index}" "$PY" -u stitch_drive.py \
    --dataset nuscenes --gpu "$GPU" --output "$OUT" \
    --val-list "$SPLIT/nuscenes_${index}.txt" \
    --checkpoint "$WEIGHT/drive_nuscenes/tcell/epoch-last.pth" &
  pids+=($!)
done
run lyft1920 "$PY" -u stitch_drive.py \
  --dataset lyft1920 --gpu "$GPU" --output "$OUT" \
  --checkpoint "$WEIGHT/drive_lyft/tcell/epoch-last.pth" &
lyft1920_pid=$!
run lyft1224 "$PY" -u stitch_drive.py \
  --dataset lyft1224 --gpu "$GPU" --output "$OUT" \
  --checkpoint "$WEIGHT/drive_lyft/tcell/epoch-last.pth" &
lyft1224_pid=$!
run widedrive3 "$PY" -u stitch_drive.py \
  --dataset widedrive3 --gpu "$GPU" --output "$OUT" \
  --checkpoint "$WEIGHT/drive_widedrive3/tcell/epoch-last.pth" &
wide_pid=$!

wait "$lyft1920_pid" "$lyft1224_pid"
echo "=== score lyft $(date '+%F %T') ===" | tee -a "$LOG/driver.log"
run score_lyft "$PY" -u eval_hm.py --dataset lyft --output "$OUT"

wait "$wide_pid"
echo "=== score widedrive3 $(date '+%F %T') ===" | tee -a "$LOG/driver.log"
run score_widedrive3 "$PY" -u eval_hm.py --dataset widedrive3 --output "$OUT"

wait "${pids[@]}"
echo "=== score nuscenes $(date '+%F %T') ===" | tee -a "$LOG/driver.log"
run score_nuscenes "$PY" -u eval_hm.py --dataset nuscenes --output "$OUT"
echo "=== finetune infer done $(date '+%F %T') ===" | tee -a "$LOG/driver.log"
