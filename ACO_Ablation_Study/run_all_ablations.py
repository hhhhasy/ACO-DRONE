"""Batch runner for the ACO+Tabu ablation study.

SAFETY
------
This script does NOT start anything unless the explicit confirmation token is
present::

    --confirm RUN-FORMAL-ABLATION

Without it, and in ``--dry-run`` / ``--check-only`` modes, no solver is ever
imported for execution and no result directory is created.  The batch is
sequential (no workers) so that every run sees the same wall-clock conditions.

Formal main experiment (30 datasets x 10 seeds x 6 variants at 50 s = 1800
runs) is written in README.md and is the user's decision to launch.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import statistics
import subprocess
import sys
from typing import Dict, List, Optional, Sequence

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
if STUDY_DIR not in sys.path:
    sys.path.insert(0, STUDY_DIR)

import ablation_config as ac  # noqa: E402
import run_state  # noqa: E402

CONFIRM_TOKEN = "RUN-FORMAL-ABLATION"
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(STUDY_DIR, "..", "大规模"))
EXIT_OK = 0
EXIT_INCOMPLETE = 1
EXIT_USAGE = 2
EXIT_CONFLICT = 4


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run the complete ACO+Tabu ablation grid. Formal execution requires "
            "--confirm %s. Use --dry-run to print commands without solving and "
            "--check-only to audit existing results." % CONFIRM_TOKEN
        )
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                     help="directory with physicalized benchmark JSON files")
    src.add_argument("--dataset", action="append", default=None,
                     help="explicit dataset JSON (repeatable); overrides --data-dir")
    p.add_argument("--pattern", default="*.json",
                   help="glob pattern inside --data-dir")
    p.add_argument("--variants", nargs="+", default=list(ac.VARIANT_KEYS),
                   choices=list(ac.VARIANT_KEYS))
    p.add_argument("--budgets", nargs="+", type=float, default=[ac.MAIN_BUDGET],
                   help="seconds; default 50 (main paper budget). "
                        "Full grid: 10 20 30 50")
    p.add_argument("--seeds", nargs="+", type=int, default=list(ac.FORMAL_SEEDS))
    p.add_argument("--ants", type=int, default=ac.ANTS)
    p.add_argument("--iterations", type=int, default=0,
                   help="0 = time budget only (official setting)")
    p.add_argument("--results-root", default=os.path.join(STUDY_DIR, "results"))
    p.add_argument("--limit-datasets", type=int, default=0,
                   help="after sorting, use at most N datasets (0 = all)")
    p.add_argument("--limit-runs", type=int, default=0,
                   help="safety cap on jobs actually launched (0 = all); "
                        "useful for a manually supervised pilot")
    p.add_argument("--on-conflict", choices=("fail", "skip", "rerun"),
                   default="fail",
                   help="what to do with an existing run whose config/code/"
                        "dataset fingerprint does not match current sources")
    p.add_argument("--dry-run", action="store_true",
                   help="print plan/commands only; never solve or create results")
    p.add_argument("--check-only", action="store_true",
                   help="audit existing results against the expected grid; never solve")
    p.add_argument("--confirm", default=None,
                   help="must equal %s to launch a real batch" % CONFIRM_TOKEN)
    return p


def _resolve_datasets(args: argparse.Namespace) -> List[str]:
    if args.dataset:
        paths = [os.path.abspath(x) for x in args.dataset]
        for path in paths:
            if not os.path.isfile(path):
                raise ValueError("dataset does not exist: %s" % path)
    else:
        paths = ac.discover_dataset_files(args.data_dir, args.pattern)
    names = [os.path.splitext(os.path.basename(path))[0] for path in paths]
    if len(set(names)) != len(names):
        raise ValueError("duplicate dataset basenames in the selection")
    if args.limit_datasets > 0:
        paths = paths[:args.limit_datasets]
    if not paths:
        raise ValueError("empty dataset selection")
    return paths


def _job_command(
    dataset_path: str,
    variant: str,
    seed: int,
    budget: float,
    args: argparse.Namespace,
    overwrite: bool,
) -> List[str]:
    cmd = [
        sys.executable,
        os.path.join(STUDY_DIR, "run_ablation.py"),
        dataset_path,
        "--variant", str(variant),
        "--seed", str(seed),
        "--time-limit", "%g" % float(budget),
        "--ants", str(args.ants),
        "--iterations", str(args.iterations),
        "--results-root", os.path.abspath(args.results_root),
        "--quiet",
    ]
    if overwrite:
        cmd.append("--overwrite")
    return cmd


def _manifest_payload(
    args: argparse.Namespace,
    datasets: Sequence[str],
    total_jobs: int,
    dry_run: bool,
) -> Dict[str, object]:
    plan = ac.summarise_plan(args.variants, args.budgets, list(datasets), args.seeds)
    return {
        "schema_version": ac.SCHEMA_VERSION,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dry_run": bool(dry_run),
        "confirm_token_required": CONFIRM_TOKEN,
        "study_dir": STUDY_DIR,
        "results_root": os.path.abspath(args.results_root),
        "default_data_dir": os.path.abspath(args.data_dir) if not args.dataset else None,
        "plan": plan,
        "total_jobs": int(total_jobs),
        "ants": int(args.ants),
        "iterations": int(args.iterations),
        "variants": {
            key: {
                "title_zh": ac.get_variant(key).title_zh,
                "description_zh": ac.get_variant(key).description_zh,
                "configuration_fingerprint": ac.variant_fingerprint(key),
                "pheromone_enabled": {
                    "edge": ac.get_variant(key).pheromone_edge,
                    "assign": ac.get_variant(key).pheromone_assign,
                    "service": ac.get_variant(key).pheromone_service,
                    "drone": ac.get_variant(key).pheromone_drone,
                },
                "service_policy_mode": ac.get_variant(key).service_policy_mode,
            }
            for key in args.variants
        },
        "code_fingerprint": ac.code_fingerprint(),
        "code_files": {
            name: ac.file_sha256(os.path.join(STUDY_DIR, name))
            for name in ac.CODE_FILES
        },
        "python": sys.version,
        "platform": sys.platform,
    }


def _write_manifest(path: str, payload: Dict[str, object]) -> None:
    tmp = path + ".tmp-%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main(argv: Optional[List[str]] = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)

    if any(b <= 0 for b in args.budgets):
        p.error("every --budgets value must be > 0")
    if args.ants <= 0:
        p.error("--ants must be > 0")
    if args.iterations < 0:
        p.error("--iterations must be >= 0")
    if args.limit_datasets < 0 or args.limit_runs < 0:
        p.error("--limit-datasets/--limit-runs must be >= 0")
    if args.dry_run and args.check_only:
        p.error("use either --dry-run or --check-only")
    for name, values in (("--variants", args.variants),
                         ("--budgets", args.budgets),
                         ("--seeds", args.seeds)):
        if len(set(values)) != len(values):
            p.error("%s contains duplicates; each value must appear once" % name)
    if args.confirm is not None and args.confirm != CONFIRM_TOKEN:
        p.error("wrong --confirm token; expected %s" % CONFIRM_TOKEN)
    if not args.dry_run and not args.check_only and args.confirm != CONFIRM_TOKEN:
        p.error(
            "formal batch not started: add --confirm %s (or use --dry-run)"
            % CONFIRM_TOKEN
        )

    # Strict single-factor design is checked before anything else.
    ac.validate_variant_isolation()

    try:
        datasets = _resolve_datasets(args)
    except ValueError as exc:
        p.error(str(exc))

    total_jobs = len(datasets) * len(args.seeds) * len(args.budgets) * len(args.variants)
    plan = ac.summarise_plan(args.variants, args.budgets, datasets, args.seeds)
    code_fp = ac.code_fingerprint()

    print("===== ACO+Tabu ablation batch =====")
    print("datasets          : %d" % len(datasets))
    print("variants          : %s" % ", ".join(args.variants))
    print("budgets (s)       : %s" % ", ".join("%g" % b for b in args.budgets))
    print("seeds             : %s" % ", ".join(str(s) for s in args.seeds))
    print("ants / iterations : %d / %d" % (args.ants, args.iterations))
    print("total jobs        : %d" % total_jobs)
    print("code fingerprint  : %s" % code_fp)
    print("results root      : %s" % os.path.abspath(args.results_root))

    # ------------------------------------------------------------------
    # Audit-only and dry-run modes: never solve, never create results.
    # ------------------------------------------------------------------
    if args.dry_run or args.check_only:
        complete = partial = missing = conflicts = 0
        rows = []
        for dataset_path in datasets:
            identity = ac.dataset_identity(dataset_path)
            for budget in args.budgets:
                for seed in args.seeds:
                    for variant in args.variants:
                        run_dir = ac.run_directory(
                            args.results_root, variant, budget,
                            str(identity["dataset"]), seed,
                        )
                        expected_run_fp = ac.run_config_fingerprint(
                            variant, seed, budget,
                            ants=args.ants, iterations=args.iterations,
                        )
                        state = run_state.inspect_run(
                            run_dir,
                            expected_run_fingerprint=expected_run_fp,
                            expected_code_fingerprint=code_fp,
                            expected_dataset_sha256=str(identity["dataset_sha256"]),
                        )
                        if state["state"] == run_state.COMPLETE:
                            complete += 1
                        elif state["state"] == run_state.MISSING:
                            missing += 1
                        elif state["state"] == run_state.PARTIAL:
                            partial += 1
                        else:
                            conflicts += 1
                        rows.append((identity["dataset"], budget, seed, variant,
                                     state["state"], state["detail"]))

        if args.check_only:
            for dataset, budget, seed, variant, state, detail in rows:
                if state != run_state.COMPLETE:
                    print("[%s] %s budget=%gs seed=%d variant=%s -- %s"
                          % (state, dataset, budget, seed, variant, detail))
            print("\ncomplete=%d missing=%d partial=%d conflicts=%d total=%d"
                  % (complete, missing, partial, conflicts, total_jobs))
            if complete == total_jobs:
                print("RESULT: expected grid is complete.")
                return EXIT_OK
            print("RESULT: result set is INCOMPLETE. No values are inferred.")
            return EXIT_INCOMPLETE

        print("\n----- DRY RUN: no solver will be executed -----")
        path_by_name = {
            str(ac.dataset_identity(path)["dataset"]): path for path in datasets
        }
        for idx, (dataset, budget, seed, variant, state, detail) in enumerate(rows[:8], 1):
            data_path = path_by_name[dataset]
            print("[%d] state=%s" % (idx, state))
            print("    " + " ".join(_job_command(
                data_path, variant, seed, budget, args, overwrite=False)))
        if len(rows) > 8:
            print("... %d more jobs not printed" % (len(rows) - 8))
        print("\nwould_launch=%d already_complete=%d missing=%d partial=%d conflicts=%d"
              % (total_jobs - complete, complete, missing, partial, conflicts))
        print("NO SOLVE EXECUTED. NO RESULTS DIRECTORY CREATED.")
        return EXIT_OK

    # ------------------------------------------------------------------
    # Formal batch (explicit token required above).
    # ------------------------------------------------------------------
    avg_budget = statistics.fmean(args.budgets)
    print("\nWARNING: this launches %d sequential solver runs; rough pure-solve "
          "time = %.1f hours at %.1f s average budget, plus instance "
          "initialisation overhead." % (total_jobs, total_jobs * avg_budget / 3600.0,
                                        avg_budget))

    results_root = os.path.abspath(args.results_root)
    os.makedirs(results_root, exist_ok=True)
    manifest_path = os.path.join(results_root, ac.BATCH_MANIFEST_FILENAME)
    manifest = _manifest_payload(args, datasets, total_jobs, dry_run=False)
    manifest["started_at_utc"] = datetime.datetime.now(
        datetime.timezone.utc).isoformat()
    manifest["status"] = "running"
    _write_manifest(manifest_path, manifest)

    launched = skipped_complete = skipped_conflict = failed = 0
    failures: List[Dict[str, object]] = []

    for dataset_path in datasets:
        identity = ac.dataset_identity(dataset_path)
        dataset_name = str(identity["dataset"])
        dataset_sha = str(identity["dataset_sha256"])
        for budget in args.budgets:
            for seed in args.seeds:
                for variant in args.variants:
                    if args.limit_runs and launched >= args.limit_runs:
                        print("\nReached --limit-runs=%d; stopping early. "
                              "Rerun the same command to resume." % args.limit_runs)
                        manifest["status"] = "stopped_by_limit"
                        manifest["completed_at_utc"] = datetime.datetime.now(
                            datetime.timezone.utc).isoformat()
                        manifest["counts"] = {
                            "launched": launched,
                            "skipped_complete": skipped_complete,
                            "skipped_conflict": skipped_conflict,
                            "failed": failed,
                        }
                        _write_manifest(manifest_path, manifest)
                        return (EXIT_OK if (failed == 0 and skipped_conflict == 0)
                                else EXIT_INCOMPLETE)

                    run_dir = ac.run_directory(
                        results_root, variant, budget, dataset_name, seed)
                    expected_run_fp = ac.run_config_fingerprint(
                        variant, seed, budget,
                        ants=args.ants, iterations=args.iterations,
                    )
                    state = run_state.inspect_run(
                        run_dir,
                        expected_run_fingerprint=expected_run_fp,
                        expected_code_fingerprint=code_fp,
                        expected_dataset_sha256=dataset_sha,
                    )
                    overwrite = False
                    if state["state"] == run_state.COMPLETE:
                        skipped_complete += 1
                        print("[resume] complete, skip: %s" % run_dir)
                        continue
                    if state["state"] in (
                        run_state.MISSING, run_state.PARTIAL,
                    ):
                        pass  # normal run / resume of an interrupted job
                    else:
                        # config / code / dataset mismatch or invalid content
                        if args.on_conflict == "skip":
                            skipped_conflict += 1
                            print("[conflict-skip] %s: %s" % (state["state"],
                                                              state["detail"]))
                            continue
                        if args.on_conflict == "fail":
                            print("[conflict] %s: %s\n           %s"
                                  % (state["state"], state["detail"], run_dir))
                            print("Stopping. Resolve with --on-conflict skip|rerun.")
                            manifest["status"] = "conflict_stop"
                            manifest["completed_at_utc"] = datetime.datetime.now(
                                datetime.timezone.utc).isoformat()
                            manifest["counts"] = {
                                "launched": launched,
                                "skipped_complete": skipped_complete,
                                "skipped_conflict": skipped_conflict,
                                "failed": failed,
                            }
                            _write_manifest(manifest_path, manifest)
                            return EXIT_CONFLICT
                        overwrite = True

                    cmd = _job_command(dataset_path, variant, seed, budget,
                                       args, overwrite=overwrite)
                    print("\n[run] %s | budget=%gs | seed=%d | %s"
                          % (dataset_name, budget, seed, variant))
                    launched += 1
                    proc = subprocess.run(cmd, cwd=STUDY_DIR)
                    if proc.returncode != 0:
                        failed += 1
                        failures.append({
                            "dataset": dataset_name, "budget": budget,
                            "seed": seed, "variant": variant,
                            "returncode": proc.returncode,
                        })
                        print("[failed] returncode=%d; continuing (resume later)."
                              % proc.returncode)
                        continue
                    # Verify the record rather than trusting the exit code alone.
                    verify = run_state.inspect_run(
                        run_dir,
                        expected_run_fingerprint=expected_run_fp,
                        expected_code_fingerprint=code_fp,
                        expected_dataset_sha256=dataset_sha,
                    )
                    if verify["state"] != run_state.COMPLETE:
                        failed += 1
                        failures.append({
                            "dataset": dataset_name, "budget": budget,
                            "seed": seed, "variant": variant,
                            "verify_state": verify["state"],
                            "detail": verify["detail"],
                        })
                        print("[failed] post-run verification: %s (%s)"
                              % (verify["state"], verify["detail"]))

    if failed == 0 and skipped_conflict == 0:
        manifest["status"] = "completed"
    elif failed == 0:
        manifest["status"] = "completed_with_skipped_conflicts"
    else:
        manifest["status"] = "completed_with_failures"
    manifest["completed_at_utc"] = datetime.datetime.now(
        datetime.timezone.utc).isoformat()
    manifest["counts"] = {
        "launched": launched,
        "skipped_complete": skipped_complete,
        "skipped_conflict": skipped_conflict,
        "failed": failed,
    }
    manifest["failures"] = failures
    _write_manifest(manifest_path, manifest)

    print("\n===== batch finished =====")
    print("launched=%d skipped_complete=%d skipped_conflict=%d failed=%d"
          % (launched, skipped_complete, skipped_conflict, failed))
    print("manifest: %s" % manifest_path)
    return EXIT_OK if (failed == 0 and skipped_conflict == 0) else EXIT_INCOMPLETE


if __name__ == "__main__":
    raise SystemExit(main())
