#!/usr/bin/env python3
"""Resumable end-to-end runner (works the same locally and on Google Colab).

    python run_pipeline.py --dataset <dataset dir or .zip> --work <work dir> --out <output dir> \
        [--mode inference|train] [--force] [--from STEP] [--package]

  --mode inference  test split only, shipped models in ./models (minutes)
  --mode train      full training + test predictions, same steps as run_all.sh (hours)

Every finished step leaves a marker in <work>/.done/, so re-running the same command after a
disconnect continues where it stopped (keep --work on Google Drive for that).  Each step's output
is streamed live and also written to <work>/logs/<step>.log.
--dataset may be the student_resource folder, its dataset/ folder, or a .zip of either; the folder
holding train/ and test/ is found automatically.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")


def find_dataset(path, work):
    """Return the directory that contains test/test_source1.tsv (extracting a zip if needed)."""
    path = os.path.abspath(os.path.expanduser(path))
    if zipfile.is_zipfile(path):
        dest = os.path.join(work, "dataset_unzipped")
        if not os.path.exists(os.path.join(dest, ".extracted")):
            print(f"extracting {path} -> {dest}", flush=True)
            with zipfile.ZipFile(path) as z:
                z.extractall(dest)
            open(os.path.join(dest, ".extracted"), "w").close()
        path = dest
    for dirpath, _, files in os.walk(path):
        if "test_source1.tsv" in files and os.path.basename(dirpath) == "test":
            return os.path.dirname(dirpath)
    sys.exit(f"ERROR: no test/test_source1.tsv found under {path}")


def steps(mode, D, W, O):
    data = f"{W}/data"
    gt = {"GT": f"{data}/train_ground_truth.parquet"}
    common = [("convert", ["convert.py", D, data], {})]
    if mode == "inference":
        return common + [
            ("prep", ["prep.py", data, W, "test_source1", "test_source2", "test_source3"], {}),
            ("blocking_test", ["run_blocking.py", "test", W], {}),
            ("features_test", ["build_features.py", "test", W], {}),
            ("submission", ["make_submission.py", W, O], {}),
        ]
    return common + [
        ("translit", ["learn_translit.py", data, "resources/indic_token_dict.json"], {}),
        ("prep", ["prep.py", data, W], {}),
        ("blocking_train", ["run_blocking.py", "train", W], {}),
        ("blocking_test", ["run_blocking.py", "test", W], {}),
        ("features_train", ["build_features.py", "train", W], gt),
        ("select_T", ["select_T.py", W], {}),
        ("features_trainT", ["build_features.py", "trainT", W], gt),
        ("features_trainRest", ["build_features.py", "trainRest", W], gt),
        ("features_test", ["build_features.py", "test", W], {}),
        ("crossfit", ["crossfit.py", W, "all"], {}),
        ("submission", ["make_submission.py", W, O], {}),
    ]


def run_step(name, args, env_extra, W):
    os.makedirs(f"{W}/logs", exist_ok=True)
    env = {**os.environ, **env_extra, "PYTHONUNBUFFERED": "1"}
    t = time.time()
    print(f"\n=== [{name}] python {' '.join(args)}", flush=True)
    with open(f"{W}/logs/{name}.log", "w") as log:
        p = subprocess.Popen([sys.executable] + args, cwd=SRC, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            if "DeprecationWarning" in line or "is ambiguous and deprecated" in line:
                continue
            sys.stdout.write(line); log.write(line)
        rc = p.wait()
    if rc != 0:
        sys.exit(f"\n!!! step '{name}' failed (exit {rc}); full log: {W}/logs/{name}.log\n"
                 f"Fix the problem and re-run the same command: finished steps are skipped.")
    print(f"=== [{name}] done in {time.time() - t:.0f}s", flush=True)


def package(D, O, pkg_dir, team):
    """Build <team>_submission.zip with the layout required by the challenge."""
    stage = os.path.join(pkg_dir, "_stage")
    shutil.rmtree(stage, ignore_errors=True)
    code = os.path.join(stage, "code", "business_entity_resolution")
    shutil.copytree(SRC, os.path.join(code, "src"), ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for f in ("README.md", "requirements.txt", "run_all.sh", "run_inference.sh", "run_pipeline.py"):
        if os.path.exists(os.path.join(ROOT, f)):
            shutil.copy(os.path.join(ROOT, f), code)
    shutil.copytree(os.path.join(ROOT, "models"), os.path.join(code, "models"))   # lets reviewers run inference directly
    os.makedirs(os.path.join(stage, "output"))
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        shutil.copy(os.path.join(O, f), os.path.join(stage, "output"))
    doc = next((os.path.join(dp, "Documentation_template.md") for base in (ROOT, os.path.dirname(D))
                for dp, _, fs in os.walk(base) if "Documentation_template.md" in fs), None)
    if doc:
        shutil.copy(doc, stage)
    else:
        print("NOTE: Documentation_template.md not found - add your filled-in copy to the zip yourself", flush=True)
    z = shutil.make_archive(os.path.join(pkg_dir, f"{team}_submission"), "zip", stage)
    shutil.rmtree(stage)
    print("submission package:", z, flush=True)
    return z


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--work", default="work")
    ap.add_argument("--out", default="output")
    ap.add_argument("--mode", choices=["inference", "train"], default="inference")
    ap.add_argument("--force", action="store_true", help="ignore markers and redo every step")
    ap.add_argument("--from", dest="from_step", default=None, help="redo this step and everything after it")
    ap.add_argument("--validate", action="store_true", help="also run the official utils/validate_submission.py (needs ~6+ GB RAM)")
    ap.add_argument("--package", action="store_true", help="build <team>_submission.zip next to the output folder")
    ap.add_argument("--team", default="team")
    a = ap.parse_args()

    W, O = os.path.abspath(a.work), os.path.abspath(a.out)
    os.makedirs(f"{W}/.done", exist_ok=True); os.makedirs(O, exist_ok=True)
    D = find_dataset(a.dataset, W)
    sys.path.insert(0, SRC)
    from common import NTHREADS
    print(f"mode={a.mode}  dataset={D}  work={W}  out={O}  threads={NTHREADS}", flush=True)

    todo = steps(a.mode, D, W, O)
    names = [n for n, _, _ in todo]
    if a.from_step and a.from_step not in names:
        sys.exit(f"--from must be one of {names}")
    redo = a.force
    t0 = time.time()
    for name, args, env in todo:
        redo = redo or name == a.from_step
        marker = f"{W}/.done/{a.mode}_{name}"
        if os.path.exists(marker) and not redo:
            print(f"--- [{name}] already done, skipping", flush=True)
            continue
        run_step(name, args, env, W)
        open(marker, "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"\nall steps finished in {time.time() - t0:.0f}s", flush=True)

    run_step("check", ["check_candidates.py", O, f"{D}/test"], {}, W)
    val = os.path.join(os.path.dirname(D), "utils", "validate_submission.py")
    if a.validate and os.path.exists(val):
        run_step("validate", [val, "--matching", f"{O}/matching_results.tsv", "--candidate",
                              f"{O}/candidate_pairs.tsv", "--test-dir", f"{D}/test"], {}, W)
    if a.package:
        package(D, O, os.path.dirname(O), a.team)
    print(f"\noutputs: {O}/matching_results.tsv  {O}/candidate_pairs.tsv", flush=True)


if __name__ == "__main__":
    main()
