#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run_all_daddy.py
Expert script to run KANMTS variations (KAN only) for ETT and Weather datasets.
This script parses existing .sh files, filters for KANMTS model, injects GPU parameters,
and executes commands one by one.

Usage:
    # Run everything
    python run_all_daddy.py

    # Skip everything before ETTh2 pred_len=336
    python run_all_daddy.py --start_dataset ETTh2 --start_pred_len 336

    # Start from Weather, all pred_lens
    python run_all_daddy.py --start_dataset Weather

    # Start from ETTm1 pred_len=192
    python run_all_daddy.py --start_dataset ETTm1 --start_pred_len 192
"""

import os
import re
import subprocess
import sys
import argparse


# ---------------------------------------------------------------------------
# Shell script parser
# ---------------------------------------------------------------------------

def parse_bash_script(file_path):
    """
    Extracts python run commands from a shell script.
    Handles variable replacement ($model_name), line continuations,
    and corrects root_path if the dataset is found in the root instead of a subdirectory.
    """
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as f:
            content = f.read()

    # Extract model_name variable if present
    model_name_match = re.search(r'model_name=(\S+)', content)
    model_name = model_name_match.group(1) if model_name_match else "KANMTS"

    # Regex to find python -u run.py blocks
    raw_commands = re.findall(
        r'python -u run\.py.*?(?=\n\n|\npython|\n$|$)', content, re.DOTALL
    )

    cleaned_commands = []
    base_dir = os.path.dirname(os.path.abspath(__file__))

    for cmd in raw_commands:
        # 1. Remove line continuation backslashes
        cmd = cmd.replace('\\\n', ' ')
        # 2. Replace variable $model_name
        cmd = cmd.replace('$model_name', model_name)

        # 3. Path correction logic
        root_path_match = re.search(r'--root_path\s+(\S+)', cmd)
        data_path_match = re.search(r'--data_path\s+(\S+)', cmd)

        if root_path_match and data_path_match:
            original_root = root_path_match.group(1).strip()
            data_file     = data_path_match.group(1).strip()

            full_path = os.path.join(
                base_dir, original_root.replace('./', ''), data_file
            )
            if not os.path.exists(full_path):
                if os.path.exists(os.path.join(base_dir, data_file)):
                    print(f"  [i] Correcting path for {data_file}: {original_root} -> ./")
                    cmd = cmd.replace(f"--root_path {original_root}", "--root_path ./")
                elif os.path.exists(os.path.join(base_dir, "dataset", data_file)):
                    print(f"  [i] Correcting path for {data_file}: {original_root} -> ./dataset/")
                    cmd = cmd.replace(f"--root_path {original_root}", "--root_path ./dataset/")

        # 4. Collapse whitespace
        cmd = ' '.join(cmd.split())

        # 5. Filter: Only run KANMTS model
        if '--model KANMTS' in cmd or '--model' not in cmd:
            if cmd:
                cleaned_commands.append(cmd)

    return cleaned_commands


# ---------------------------------------------------------------------------
# Dataset order — used to decide what comes "before" the start point
# ---------------------------------------------------------------------------

# Edit this list to match the order your .sh scripts naturally produce commands in.
DATASET_ORDER = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather']


def _dataset_of(cmd):
    """Return the dataset name found in a command string, or None."""
    for ds in DATASET_ORDER:
        if f"--data {ds}" in cmd:
            return ds
    return None


def _pred_len_of(cmd):
    """Return the pred_len integer found in a command string, or None."""
    m = re.search(r'--pred_len\s+(\d+)', cmd)
    return int(m.group(1)) if m else None


def _should_skip(cmd, start_dataset, start_pred_len, found_start):
    """
    Returns True if this command should be skipped based on the start point.

    Logic:
      - If no start_dataset is set, never skip.
      - Skip all commands whose dataset comes before start_dataset in DATASET_ORDER.
      - For the start_dataset itself, skip pred_lens strictly less than start_pred_len
        (if start_pred_len is set).
      - Once found_start is True, never skip.
    """
    if found_start:
        return False
    if start_dataset is None:
        return False

    ds = _dataset_of(cmd)

    # Command has no recognisable dataset tag — don't skip it
    if ds is None:
        return False

    ds_idx    = DATASET_ORDER.index(ds)          if ds             in DATASET_ORDER else -1
    start_idx = DATASET_ORDER.index(start_dataset) if start_dataset in DATASET_ORDER else -1

    # Dataset comes before the start dataset → skip
    if ds_idx < start_idx:
        return True

    # Dataset comes after the start dataset → don't skip
    if ds_idx > start_idx:
        return False

    # Same dataset as start_dataset
    if start_pred_len is None:
        return False   # no pred_len filter, start from first command of this dataset

    pl = _pred_len_of(cmd)
    if pl is None:
        return False

    return pl < start_pred_len


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

def execute_commands(
    commands,
    gpu_args="--use_gpu 1 --gpu 0",
    auto_continue=True,
    cwd=None,
    start_dataset=None,
    start_pred_len=None,
):
    """
    Executes a list of commands sequentially, optionally skipping
    everything before (start_dataset, start_pred_len).
    """
    if not commands:
        print("\n[!] No commands to execute.")
        return

    summary_file_path = os.path.join(cwd if cwd else ".", "experiments_summary.txt")
    results_file_path = os.path.join(cwd if cwd else ".", "result", "results.txt")

    # ── Apply skip logic ──────────────────────────────────────────────────
    found_start   = False
    skipped_count = 0
    kept_commands = []

    for cmd in commands:
        if _should_skip(cmd, start_dataset, start_pred_len, found_start):
            ds = _dataset_of(cmd) or "unknown"
            pl = _pred_len_of(cmd) or "?"
            print(f"  [SKIP] {ds} pred_len={pl}  →  {cmd[:70]}...")
            skipped_count += 1
        else:
            found_start = True   # once we keep one, stop skipping
            kept_commands.append(cmd)

    print(f"\n  Skipped : {skipped_count} command(s)")
    print(f"  Running : {len(kept_commands)} command(s)")
    # ─────────────────────────────────────────────────────────────────────

    total = len(kept_commands)
    if total == 0:
        print("\n[!] Nothing left to run after applying skip filter.")
        return

    print(f"\nStarting execution of {total} experiments...")
    print(f"Summary will be written to: {summary_file_path}\n")

    for i, cmd in enumerate(kept_commands, 1):
        # Inject GPU args if missing
        final_cmd = cmd
        if "--use_gpu" not in final_cmd:
            final_cmd += f" {gpu_args}"

        print(f"\n[{i}/{total}] Executing:")
        print(f"  {final_cmd}")
        print("-" * 80)

        try:
            subprocess.run(final_cmd, shell=True, check=True, cwd=cwd)

            # Extract summary from results.txt
            if os.path.exists(results_file_path):
                try:
                    with open(results_file_path, "r", encoding="utf-8", errors="replace") as f:
                        lines = f.readlines()

                    summary_lines = []
                    summary_keys  = ["MSE:", "MAE:", "parameters:", "time:", "Memory:", "Peak"]

                    for line in reversed(lines):
                        if "setting:" in line:
                            summary_lines.insert(0, line)
                            break
                        if any(key in line for key in summary_keys):
                            summary_lines.insert(0, line)

                    if summary_lines:
                        with open(summary_file_path, "a", encoding="utf-8") as sf:
                            sf.write("\n" + "=" * 80 + "\n")
                            sf.writelines(summary_lines)
                        print(f"  [+] Summary saved to {os.path.basename(summary_file_path)}")

                except Exception as e:
                    print(f"  [!] Could not update summary file: {e}")

        except subprocess.CalledProcessError as e:
            print(f"\n[!] Command failed with exit code {e.returncode}")
            if not auto_continue:
                cont = input("Continue to next experiment? (y/n): ")
                if cont.lower() != 'y':
                    print("Aborting.")
                    sys.exit(1)
            else:
                print("Auto-continuing to next experiment...")

        except KeyboardInterrupt:
            print("\n[!] Interrupted by user. Exiting...")
            sys.exit(0)

    print("\n" + "=" * 40)
    print("All experiments completed!")
    print("=" * 40)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run KANMTS experiments from .sh scripts, with optional start point."
    )
    parser.add_argument(
        '--start_dataset',
        type=str,
        default=None,
        choices=DATASET_ORDER,
        help=(
            'Dataset to start from. Everything before this dataset is skipped. '
            f'Options: {DATASET_ORDER}'
        )
    )
    parser.add_argument(
        '--start_pred_len',
        type=int,
        default=None,
        help=(
            'Prediction length to start from within --start_dataset. '
            'Commands for that dataset with pred_len < this value are skipped. '
            'Example: --start_dataset ETTh2 --start_pred_len 336'
        )
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU index to use (default: 0)'
    )
    parser.add_argument(
        '--no_auto_continue',
        action='store_true',
        help='Pause and ask before continuing after a failed experiment'
    )
    args = parser.parse_args()

    # Validate
    if args.start_pred_len is not None and args.start_dataset is None:
        parser.error("--start_pred_len requires --start_dataset to be set.")

    if args.start_dataset:
        msg = f"Starting from dataset={args.start_dataset}"
        if args.start_pred_len:
            msg += f", pred_len={args.start_pred_len}"
        print(f"\n[i] {msg}")
        print(f"[i] All experiments before this point will be skipped.\n")

    base_dir    = os.path.dirname(os.path.abspath(__file__))
    scripts_root = os.path.join(base_dir, "scripts")
    all_cmds    = []

    print(f"Searching for .sh scripts in: {scripts_root}")
    if os.path.exists(scripts_root):
        for root, dirs, files in os.walk(scripts_root):
            if "financial" in root.lower():
                continue
            for f in sorted(files):
                if f.endswith(".sh") and "financial" not in f.lower():
                    script_path = os.path.join(root, f)
                    cmds = parse_bash_script(script_path)
                    if cmds:
                        print(f"  - Found {len(cmds)} command(s) in "
                              f"{os.path.relpath(script_path, scripts_root)}")
                        all_cmds.extend(cmds)
    else:
        print(f"Error: Scripts directory not found at {scripts_root}")
        sys.exit(1)

    if not all_cmds:
        print("\n[!] No commands found. Please verify the directory structure.")
        return

    execute_commands(
        commands      = all_cmds,
        gpu_args      = f"--use_gpu 1 --gpu {args.gpu}",
        auto_continue = not args.no_auto_continue,
        cwd           = base_dir,
        start_dataset = args.start_dataset,
        start_pred_len= args.start_pred_len,
    )


if __name__ == "__main__":
    main()
