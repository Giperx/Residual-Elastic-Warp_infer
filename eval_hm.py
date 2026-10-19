#!/usr/bin/env python
"""Repository entry point for the driving HM metrics.

    python eval_hm.py --dataset nuscenes
    python eval_hm.py --dataset lyft

Scores every frame under output/metric_eval/{dataset}/rewarp and writes
output/metric_eval/{dataset}/rewarp_hm.txt. No torch and no LPIPS.
"""
from rewarp_drive.metrics import main

if __name__ == "__main__":
    raise SystemExit(main())
