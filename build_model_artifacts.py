from __future__ import annotations

import argparse
import gzip
import json
import pickle
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from icpm_experiments import add_transition_mappings_to_model, prepare_model
from run_dice_memo_comparison import (
    ExperimentRun,
    build_runs,
    count_trace_variants,
    load_config,
    split_log_once,
    validate_run_paths,
)
from utils import generate_model_from_file


def now_epoch() -> float:
    return time.time()


def git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def normalize_log_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["case:concept:name"] = result["case:concept:name"].astype(str)
    result["concept:name"] = result["concept:name"].astype(str)
    return result


def build_model(run: ExperimentRun, train_df: pd.DataFrame, map_dict: Dict[Any, str]) -> Any:
    if run.read_model_from_file:
        if not run.model_path:
            raise ValueError(f"{run.run_id} reads a model from file but has no model_path")
        model = generate_model_from_file(
            run.model_path,
            activity_mapping_dict=map_dict,
            return_markings=False,
        )
    else:
        model = prepare_model(train_df, run.non_sync_penalty)

    if run.use_heuristics:
        model = add_transition_mappings_to_model(model)
    return model


def artifact_metadata(
    run: ExperimentRun,
    model: Any,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    elapsed_seconds: float,
) -> Dict[str, Any]:
    return {
        "artifact_format": "dice_model_pickle_gzip_v1",
        "git_commit": git_commit(),
        "run_id": run.run_id,
        "dataset": run.dataset_name,
        "category": run.category,
        "trace_split": run.split_name or "all",
        "window_len": run.window_len,
        "source": "reference_model_file" if run.read_model_from_file else "discovered_from_train_split",
        "model_path": run.model_path,
        "non_sync_penalty": run.non_sync_penalty,
        "use_heuristics": run.use_heuristics,
        "n_places": len(model.places),
        "n_transitions": len(model.transitions),
        "n_mandatory_markings": len(model.mandatory_transitions_map or {}),
        "n_alive_markings": len(model.alive_transitions_map or {}),
        "n_filtered_cases": int(filtered_df["case:concept:name"].nunique()),
        "n_filtered_trace_variants": count_trace_variants(filtered_df),
        "n_train_cases": int(train_df["case:concept:name"].nunique()),
        "n_train_trace_variants": count_trace_variants(train_df),
        "n_test_cases": int(test_df["case:concept:name"].nunique()),
        "n_test_trace_variants": count_trace_variants(test_df),
        "build_elapsed_seconds": elapsed_seconds,
    }


def write_artifact(run: ExperimentRun, force: bool) -> Optional[Path]:
    if not run.model_artifact_path:
        raise ValueError(f"{run.run_id} has no model_artifact_path configured")

    artifact_path = Path(run.model_artifact_path)
    metadata_path = artifact_path.with_name("metadata.json")
    if artifact_path.exists() and metadata_path.exists() and not force:
        print(f"Skipping existing artifact: {run.run_id}")
        return artifact_path

    validate_run_paths(run)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    train_df, test_df, map_dict, filtered_df = split_log_once(run)
    train_df = normalize_log_columns(train_df)
    test_df = normalize_log_columns(test_df)
    filtered_df = normalize_log_columns(filtered_df)

    print(f"Building model artifact: {run.run_id}")
    start = now_epoch()
    model = build_model(run, train_df, map_dict)
    elapsed = now_epoch() - start

    with gzip.open(artifact_path, "wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    metadata = artifact_metadata(run, model, train_df, test_df, filtered_df, elapsed)
    metadata["artifact_path"] = str(artifact_path)
    metadata["artifact_size_bytes"] = artifact_path.stat().st_size
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"Wrote {artifact_path} ({artifact_path.stat().st_size} bytes)")
    return artifact_path


def load_artifact(path: Path) -> Any:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build precomputed model artifacts for DICE reproduction runs.")
    parser.add_argument("--config", type=Path, default=Path("configs/dice_memo_comparison.toml"))
    parser.add_argument("--run-id", action="append", help="Build only a specific run_id. Can be provided multiple times.")
    parser.add_argument("--limit-runs", type=int, help="Build only the first N selected runs.")
    parser.add_argument("--force", action="store_true", help="Rebuild artifacts even if they already exist.")
    parser.add_argument("--list-runs", action="store_true", help="List artifact paths and exit.")
    parser.add_argument("--validate-load", action="store_true", help="Load each written artifact after saving it.")
    return parser.parse_args()


def select_runs(runs: List[ExperimentRun], requested_ids: Optional[List[str]], limit_runs: Optional[int]) -> List[ExperimentRun]:
    selected = runs
    if requested_ids:
        requested = set(requested_ids)
        selected = [run for run in selected if run.run_id in requested]
        missing = sorted(requested - {run.run_id for run in selected})
        if missing:
            raise ValueError(f"Unknown run_id(s): {', '.join(missing)}")
    if limit_runs is not None:
        selected = selected[:limit_runs]
    return selected


def main() -> int:
    args = parse_args()
    runs = select_runs(build_runs(load_config(args.config)), args.run_id, args.limit_runs)

    if args.list_runs:
        for run in runs:
            print(f"{run.run_id}: {run.model_artifact_path}")
        return 0

    for index, run in enumerate(runs, start=1):
        print(f"\n[{index}/{len(runs)}] {run.run_id}")
        artifact_path = write_artifact(run, force=args.force)
        if args.validate_load and artifact_path is not None:
            loaded = load_artifact(artifact_path)
            print(
                "Validated load: "
                f"places={len(loaded.places)} transitions={len(loaded.transitions)} "
                f"mandatory={len(loaded.mandatory_transitions_map or {})} "
                f"alive={len(loaded.alive_transitions_map or {})}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
