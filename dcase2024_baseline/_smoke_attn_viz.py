"""Single-file CPU smoke test for the attention viz pipeline.

Loads cells 110/111/112 seed 0, runs forward on ONE STARSS23 dev-test file
on CPU, and plots. Only used to validate the pipeline before the full
multi-file run (which is fast on GPU).
"""
import sys, os
sys.path.insert(0, r"D:\ssl-research\dcase2024_baseline")
sys.path.insert(0, r"D:\ssl-research")

# Pin to CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import _path_c_attn_viz as viz

# Override target list to just one file for smoke
viz.TARGET_FILES = ["fold4_room23_mix001"]
sys.exit(viz.main())
