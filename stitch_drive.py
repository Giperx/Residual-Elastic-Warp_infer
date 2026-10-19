#!/usr/bin/env python
"""Repository entry point for the driving stitches.

    python stitch_drive.py --dataset nuscenes --scene 002 --frame 0
    python stitch_drive.py --dataset lyft1920 --limit 2

Writes the full stitch and the camera-5 crop resized to the sparse GT.
Masks are not written. Failed frames write neither file.

Importing ``rewarp_drive.drive`` parses ``--gpu`` from argv and sets
``CUDA_VISIBLE_DEVICES`` before torch is imported.
"""
from rewarp_drive.drive import main

if __name__ == "__main__":
    raise SystemExit(main())
