"""Stand-alone test-only evaluator for an existing best.pt produced by train_seldnet.

Usage:
    python test_only.py <task_id> <job_id> [<model_path>]

If model_path is omitted, falls back to the canonical
{model_dir}/{task}_{job}_dev_split0_multiaccdoa_{feat}_model.h5 path used by train_seldnet.
"""
from __future__ import annotations

import os
import sys
from time import gmtime, strftime

import numpy as np
import torch
import torch.nn as nn

import cls_data_generator
import cls_feature_class
import parameters
import seldnet_model
from cls_compute_seld_results import ComputeSELDResults


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("usage: python test_only.py <task_id> <job_id> [<model_path>]")
        return 1

    task_id = argv[1]
    job_id = argv[2]
    explicit_model = argv[3] if len(argv) >= 4 else None

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"[test_only] torch = {torch.__version__} cuda_runtime = {torch.version.cuda}")
    print(f"[test_only] device = {device}")

    params = parameters.get_params(task_id)

    if "2024" in params["dataset_dir"]:
        test_splits = [[4]]
    elif "2023" in params["dataset_dir"]:
        test_splits = [[4]]
    else:
        print("ERROR: only supports 2023/2024 dataset paths in test_only.py")
        return 2

    feat_tag = "foa" if params["dataset"] == "foa" else (
        "mic_salsa" if params.get("use_salsalite") else "mic_gcc"
    )
    out_tag = "multiaccdoa" if params["multi_accdoa"] else "accdoa"
    unique_name = f"{task_id}_{job_id}_dev_split0_{out_tag}_{feat_tag}"
    default_model_name = os.path.join(params["model_dir"], f"{unique_name}_model.h5")
    model_name = explicit_model or default_model_name

    if not os.path.isfile(model_name):
        print(f"ERROR: best.pt not found at {model_name}")
        return 3

    print(f"[test_only] checkpoint = {model_name}")

    data_gen_test = cls_data_generator.DataGenerator(
        params=params, split=test_splits[0], shuffle=False, per_file=True
    )
    filelist = data_gen_test.get_filelist()
    digest = data_gen_test.get_file_order_digest() if hasattr(data_gen_test, "get_file_order_digest") else "na"
    first = filelist[0] if filelist else "na"
    last = filelist[-1] if filelist else "na"
    print(f"[test_only] test files = {len(filelist)} digest = {digest} first = {first} last = {last}")
    data_in, data_out = data_gen_test.get_data_sizes()
    model = seldnet_model.SeldModel(data_in, data_out, params).to(device)
    state_dict = torch.load(model_name, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    dcase_output_test_folder = os.path.join(
        params["dcase_output_dir"],
        f"{unique_name}_{strftime('%Y%m%d%H%M%S', gmtime())}_test_only",
    )
    cls_feature_class.delete_and_create_folder(dcase_output_test_folder)
    print(f"[test_only] dump dir   = {dcase_output_test_folder}")

    if params["multi_accdoa"]:
        criterion = seldnet_model.MSELoss_ADPIT()
    else:
        criterion = nn.MSELoss()

    from train_seldnet import test_epoch  # reuse exact inference logic
    test_loss = test_epoch(
        data_gen_test, model, criterion, dcase_output_test_folder, params, device
    )

    score_obj = ComputeSELDResults(params)
    use_jackknife = True
    (
        test_ER, test_F, test_LE,
        test_dist_err, test_rel_dist_err,
        test_LR, test_seld_scr, classwise_test_scr,
    ) = score_obj.get_SELD_Results(
        dcase_output_test_folder, is_jackknife=use_jackknife
    )

    print()
    print("=" * 72)
    print(f"  TEST EVAL on STARSS23 dev-test  (jackknife 95% CI)")
    print("=" * 72)
    seld = test_seld_scr[0] if use_jackknife else test_seld_scr
    seld_ci = test_seld_scr[1] if use_jackknife else (None, None)
    f1 = test_F[0] if use_jackknife else test_F
    f1_ci = test_F[1] if use_jackknife else (None, None)
    le = test_LE[0] if use_jackknife else test_LE
    le_ci = test_LE[1] if use_jackknife else (None, None)
    rde = test_rel_dist_err[0] if use_jackknife else test_rel_dist_err
    rde_ci = test_rel_dist_err[1] if use_jackknife else (None, None)
    de = test_dist_err[0] if use_jackknife else test_dist_err
    de_ci = test_dist_err[1] if use_jackknife else (None, None)

    print(f"  SELD score (early-stopping)   : {seld:0.3f}    [{seld_ci[0]:.3f}, {seld_ci[1]:.3f}]")
    print(f"  F 20° (location-aware F1)    : {100*f1:0.2f} %  [{100*f1_ci[0]:.2f} %, {100*f1_ci[1]:.2f} %]")
    print(f"  DOAE_CD (deg)                : {le:0.2f}    [{le_ci[0]:.2f}, {le_ci[1]:.2f}]")
    print(f"  Dist_err (m)                 : {de:0.2f}    [{de_ci[0]:.2f}, {de_ci[1]:.2f}]")
    print(f"  RDE_CD (rel)                 : {rde:0.2f}    [{rde_ci[0]:.2f}, {rde_ci[1]:.2f}]")
    print()
    print(f"  Reference (DCASE 2024 README, FOA Multi-ACCDDOA):")
    print(f"    F 20° = 13.1 %, DOAE_CD = 36.9°, RDE = 0.33")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
