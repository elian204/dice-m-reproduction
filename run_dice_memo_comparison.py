from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from icpm_experiments import (
    compare_window_based_baselines,
    get_case_variants,
    load_and_preprocess_log,
    select_unique_variant_cases,
    train_test_log_split_simplified,
)


@dataclass(frozen=True)
class ExperimentRun:
    run_id: str
    dataset_name: str
    category: str
    split_name: Optional[str]
    df_name: str
    data_path: str
    subfolder: str
    window_len: int
    min_len: Optional[int]
    max_len: Optional[int]
    n_train_traces: int
    n_test_traces: Optional[int]
    train_cases: Optional[List[str]]
    test_cases: Optional[List[str]]
    n_traces: Optional[int]
    random_seed: int
    n_final_markings_lst: List[int]
    window_overlap: int
    use_heuristics: bool
    nonsync_density_tolerance: float
    portion: float
    max_successive_merges: int
    non_sync_penalty: int
    allow_variant_intersection: bool
    unique_train_variants: bool
    unique_test_variants: bool
    max_samples_per_activity: Optional[int]
    read_model_from_file: bool
    model_path: Optional[str]


def slugify(value: str) -> str:
    value = value.strip().replace("/", "-")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_")


def optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def optional_str_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def build_runs(config: Dict[str, Any], max_input_traces: Optional[int] = None) -> List[ExperimentRun]:
    paths = config["paths"]
    defaults = config["defaults"]
    splits = config.get("trace_length_splits", {})
    runs: List[ExperimentRun] = []

    for dataset in config["datasets"]:
        split_items: Iterable[tuple[Optional[str], Dict[str, Any]]]
        if dataset.get("apply_trace_length_splits", False):
            split_items = splits.items()
        else:
            split_items = [(None, {})]

        for split_name, split_cfg in split_items:
            for window_len in dataset.get("window_lengths_lst", defaults.get("window_lengths_lst", [])):
                n_traces = dataset.get("n_traces", defaults.get("n_traces"))
                if max_input_traces is not None:
                    n_traces = max_input_traces

                run_name_parts = [
                    dataset["name"],
                    split_name or "all",
                    f"w{window_len}",
                ]
                run_id = slugify("__".join(run_name_parts))

                runs.append(
                    ExperimentRun(
                        run_id=run_id,
                        dataset_name=dataset["name"],
                        category=dataset["category"],
                        split_name=split_name,
                        df_name=dataset["df_name"],
                        data_path=str(paths["data_root"]),
                        subfolder=dataset["subfolder"],
                        window_len=int(window_len),
                        min_len=optional_int(dataset.get("min_len", split_cfg.get("min_len", defaults.get("min_len")))),
                        max_len=optional_int(dataset.get("max_len", split_cfg.get("max_len", defaults.get("max_len")))),
                        n_train_traces=int(dataset.get("n_train_traces", defaults["n_train_traces"])),
                        n_test_traces=optional_int(dataset.get("n_test_traces", defaults.get("n_test_traces"))),
                        train_cases=optional_str_list(dataset.get("train_cases", defaults.get("train_cases"))),
                        test_cases=optional_str_list(dataset.get("test_cases", defaults.get("test_cases"))),
                        n_traces=optional_int(n_traces),
                        random_seed=int(dataset.get("random_seed", defaults["random_seed"])),
                        n_final_markings_lst=list(dataset.get("n_final_markings_lst", defaults["n_final_markings_lst"])),
                        window_overlap=int(dataset.get("window_overlap", defaults["window_overlap"])),
                        use_heuristics=bool(dataset.get("use_heuristics", defaults["use_heuristics"])),
                        nonsync_density_tolerance=float(
                            dataset.get("nonsync_density_tolerance", defaults["nonsync_density_tolerance"])
                        ),
                        portion=float(dataset.get("portion", defaults["portion"])),
                        max_successive_merges=int(
                            dataset.get("max_successive_merges", defaults["max_successive_merges"])
                        ),
                        non_sync_penalty=int(dataset.get("non_sync_penalty", defaults["non_sync_penalty"])),
                        allow_variant_intersection=bool(
                            dataset.get("allow_variant_intersection", defaults["allow_variant_intersection"])
                        ),
                        unique_train_variants=bool(
                            dataset.get("unique_train_variants", defaults["unique_train_variants"])
                        ),
                        unique_test_variants=bool(
                            dataset.get("unique_test_variants", defaults["unique_test_variants"])
                        ),
                        max_samples_per_activity=optional_int(
                            dataset.get("max_samples_per_activity", defaults.get("max_samples_per_activity"))
                        ),
                        read_model_from_file=bool(dataset.get("read_model_from_file", False)),
                        model_path=dataset.get("model_path"),
                    )
                )
    return runs


def validate_run_paths(run: ExperimentRun) -> None:
    csv_path = Path(run.data_path) / run.subfolder / run.df_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found for {run.run_id}: {csv_path}")
    if run.read_model_from_file:
        if not run.model_path:
            raise ValueError(f"{run.run_id} reads a model from file but model_path is empty")
        if not Path(run.model_path).exists():
            raise FileNotFoundError(f"Model file not found for {run.run_id}: {run.model_path}")


def split_by_fixed_cases(
    df: pd.DataFrame,
    run: ExperimentRun,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    variant_df = df.copy()
    variant_df["case:concept:name"] = variant_df["case:concept:name"].astype(str)
    case_series = variant_df["case:concept:name"]
    available_cases = list(dict.fromkeys(case_series.tolist()))
    available_case_set = set(available_cases)
    case_variants = get_case_variants(variant_df)

    if run.test_cases is None:
        test_cases = None
    else:
        test_cases = [str(case) for case in run.test_cases]
        missing_test = sorted(set(test_cases) - available_case_set)
        if missing_test:
            raise ValueError(f"{run.run_id} test_cases not found after filtering: {missing_test}")

    if run.train_cases is None:
        if test_cases is None:
            raise ValueError(f"{run.run_id} has fixed-case splitting enabled without train_cases or test_cases")
        candidate_train_cases = [case for case in available_cases if case not in set(test_cases)]
        if run.n_train_traces > len(candidate_train_cases):
            raise ValueError(
                f"{run.run_id} requested {run.n_train_traces} train cases but only "
                f"{len(candidate_train_cases)} are available outside test_cases"
            )
        train_cases = random.Random(run.random_seed).sample(candidate_train_cases, run.n_train_traces)
    else:
        train_cases = [str(case) for case in run.train_cases]
        missing_train = sorted(set(train_cases) - available_case_set)
        if missing_train:
            raise ValueError(f"{run.run_id} train_cases not found after filtering: {missing_train}")
        if run.unique_train_variants:
            train_cases = select_unique_variant_cases(train_cases, case_variants, random_seed=run.random_seed)

    if test_cases is None:
        train_case_set = set(train_cases)
        test_cases = [case for case in available_cases if case not in train_case_set]

    if not run.allow_variant_intersection:
        train_variants = {case_variants[case] for case in train_cases}
        test_cases = [case for case in test_cases if case_variants[case] not in train_variants]

    if run.unique_test_variants:
        test_cases = select_unique_variant_cases(test_cases, case_variants, random_seed=run.random_seed)

    overlap = sorted(set(train_cases) & set(test_cases))
    if overlap:
        raise ValueError(f"{run.run_id} has cases in both train_cases and test_cases: {overlap}")

    train_df = df[case_series.isin(set(train_cases))].copy()
    test_df = df[case_series.isin(set(test_cases))].copy()
    return train_df, test_df


def split_log_once(run: ExperimentRun) -> tuple[pd.DataFrame, pd.DataFrame, Dict[Any, str], pd.DataFrame]:
    df, map_dict = load_and_preprocess_log(
        run.df_name,
        min_len=run.min_len,
        max_len=run.max_len,
        n_traces=run.n_traces,
        random_seed=run.random_seed,
        path=run.data_path,
        subfolder=run.subfolder,
        max_samples_per_activity=run.max_samples_per_activity,
        stats=None,
    )
    if run.train_cases is not None or run.test_cases is not None:
        train_df, test_df = split_by_fixed_cases(df, run)
        return train_df, test_df, map_dict, df

    split = train_test_log_split_simplified(
        df,
        n_train_traces=run.n_train_traces,
        n_test_traces=run.n_test_traces,
        random_seed=run.random_seed,
        allow_variant_intersection=run.allow_variant_intersection,
        unique_train_variants=run.unique_train_variants,
        unique_test_variants=run.unique_test_variants,
    )
    return split["train_df"], split["test_df"], map_dict, df


def count_trace_variants(df: pd.DataFrame) -> int:
    variant_df = df.copy()
    variant_df["case:concept:name"] = variant_df["case:concept:name"].astype(str)
    return len(set(get_case_variants(variant_df).values()))


def run_variant(
    run: ExperimentRun,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    map_dict: Dict[Any, str],
    use_memo: bool,
) -> tuple[pd.DataFrame, Dict[str, Dict[str, list]], float]:
    start = time.time()
    summary_df, _model, raw_results = compare_window_based_baselines(
        train_df=train_df,
        test_df=test_df,
        window_lengths_lst=[run.window_len],
        n_final_markings_lst=run.n_final_markings_lst,
        window_overlap=run.window_overlap,
        read_model_from_file=run.read_model_from_file,
        model_path=run.model_path or "",
        non_sync_penalty=run.non_sync_penalty,
        allow_variant_intersection=run.allow_variant_intersection,
        cost_function=None,
        use_heuristics=run.use_heuristics,
        random_seed=run.random_seed,
        map_dict=map_dict,
        print_dataset_stats=False,
        return_test_df=False,
        portion=run.portion,
        nonsync_density_tolerance=run.nonsync_density_tolerance,
        use_memo=use_memo,
        max_successive_merges=run.max_successive_merges,
    )
    return summary_df, raw_results, time.time() - start


def raw_per_case_metrics(raw_results: Dict[str, Dict[str, list]], window_len: int) -> pd.DataFrame:
    key = next(iter(raw_results))
    data = raw_results[key]
    return pd.DataFrame(
        {
            "case_id": data["case_id"],
            f"window_{window_len}_cost": data["cost"],
            f"window_{window_len}_time": data["time"],
            f"window_{window_len}_nodes_opened": data["nodes_opened"],
        }
    )


def add_trace_lengths(df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    lengths = test_df.groupby("case:concept:name").size().rename("trace_length").reset_index()
    lengths = lengths.rename(columns={"case:concept:name": "case_id"})
    lengths["case_id"] = lengths["case_id"].astype(str)
    result = df.copy()
    result["case_id"] = result["case_id"].astype(str)
    return result.merge(lengths, on="case_id", how="left")


def compare_pair(
    run: ExperimentRun,
    dice_metrics: pd.DataFrame,
    memo_metrics: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    cost_col = f"window_{run.window_len}_cost"
    time_col = f"window_{run.window_len}_time"
    nodes_col = f"window_{run.window_len}_nodes_opened"

    dice = dice_metrics.rename(
        columns={
            cost_col: "dice_cost",
            time_col: "dice_time",
            nodes_col: "dice_nodes_opened",
        }
    )
    memo = memo_metrics.rename(
        columns={
            cost_col: "memo_cost",
            time_col: "memo_time",
            nodes_col: "memo_nodes_opened",
        }
    )
    comparison = dice.merge(memo, on="case_id", how="outer")
    comparison = add_trace_lengths(comparison, test_df)
    comparison.insert(0, "run_id", run.run_id)
    comparison.insert(1, "dataset", run.dataset_name)
    comparison.insert(2, "category", run.category)
    comparison.insert(3, "trace_split", run.split_name or "all")
    comparison.insert(4, "window_len", run.window_len)

    comparison["cost_delta"] = comparison["memo_cost"] - comparison["dice_cost"]
    comparison["abs_cost_delta"] = comparison["cost_delta"].abs()
    comparison["relative_cost_delta"] = comparison.apply(
        lambda row: math.nan
        if row["dice_cost"] == 0
        else (row["memo_cost"] - row["dice_cost"]) / row["dice_cost"],
        axis=1,
    )
    comparison["costs_equal"] = comparison["abs_cost_delta"].fillna(math.inf) <= 1e-9
    comparison["time_delta"] = comparison["memo_time"] - comparison["dice_time"]
    comparison["nodes_delta"] = comparison["memo_nodes_opened"] - comparison["dice_nodes_opened"]
    return comparison


def aggregate_comparison(run: ExperimentRun, comparison: pd.DataFrame, dice_wall: float, memo_wall: float) -> Dict[str, Any]:
    return {
        "run_id": run.run_id,
        "dataset": run.dataset_name,
        "category": run.category,
        "trace_split": run.split_name or "all",
        "window_len": run.window_len,
        "n_cases": int(len(comparison)),
        "n_cost_mismatches": int((~comparison["costs_equal"]).sum()),
        "cost_match_rate": float(comparison["costs_equal"].mean()) if len(comparison) else math.nan,
        "max_abs_cost_delta": float(comparison["abs_cost_delta"].max()) if len(comparison) else math.nan,
        "mean_abs_cost_delta": float(comparison["abs_cost_delta"].mean()) if len(comparison) else math.nan,
        "mean_cost_delta": float(comparison["cost_delta"].mean()) if len(comparison) else math.nan,
        "mean_dice_cost": float(comparison["dice_cost"].mean()) if len(comparison) else math.nan,
        "mean_memo_cost": float(comparison["memo_cost"].mean()) if len(comparison) else math.nan,
        "mean_dice_time": float(comparison["dice_time"].mean()) if len(comparison) else math.nan,
        "mean_memo_time": float(comparison["memo_time"].mean()) if len(comparison) else math.nan,
        "mean_dice_nodes_opened": float(comparison["dice_nodes_opened"].mean()) if len(comparison) else math.nan,
        "mean_memo_nodes_opened": float(comparison["memo_nodes_opened"].mean()) if len(comparison) else math.nan,
        "dice_wall_time": dice_wall,
        "memo_wall_time": memo_wall,
        "skipped_existing": False,
    }


def load_completed_aggregate(run_dir: Path) -> Optional[Dict[str, Any]]:
    aggregate_path = run_dir / "aggregate.json"
    comparison_path = run_dir / "comparison.csv"
    if not aggregate_path.exists() or not comparison_path.exists():
        return None
    aggregate = json.loads(aggregate_path.read_text())
    aggregate["skipped_existing"] = True
    return aggregate


def run_experiment(
    run: ExperimentRun,
    output_root: Path,
    save_summary_csvs: bool,
    save_per_trace_results: bool,
    save_full_alignments: bool,
    force_rerun: bool,
) -> Dict[str, Any]:
    validate_run_paths(run)
    run_dir = output_root / run.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if not force_rerun:
        completed = load_completed_aggregate(run_dir)
        if completed is not None:
            print(f"Skipping completed run: {run.run_id}")
            return completed

    train_df, test_df, map_dict, filtered_df = split_log_once(run)
    if test_df.empty:
        raise ValueError(f"{run.run_id} produced an empty test split")

    metadata = {
        **run.__dict__,
        "split_mode": "fixed_cases" if (run.train_cases is not None or run.test_cases is not None) else "random_seed",
        "n_filtered_cases": int(filtered_df["case:concept:name"].nunique()),
        "n_filtered_events": int(len(filtered_df)),
        "n_train_cases": int(train_df["case:concept:name"].nunique()),
        "n_test_cases": int(test_df["case:concept:name"].nunique()),
        "n_filtered_trace_variants": count_trace_variants(filtered_df),
        "n_train_trace_variants": count_trace_variants(train_df),
        "n_test_trace_variants": count_trace_variants(test_df),
        "n_train_events": int(len(train_df)),
        "n_test_events": int(len(test_df)),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    train_df.to_csv(run_dir / "train_split.csv", index=False)
    test_df.to_csv(run_dir / "test_split.csv", index=False)

    dice_summary, dice_raw, dice_wall = run_variant(run, train_df, test_df, map_dict, use_memo=False)
    memo_summary, memo_raw, memo_wall = run_variant(run, train_df, test_df, map_dict, use_memo=True)

    dice_metrics = raw_per_case_metrics(dice_raw, run.window_len)
    memo_metrics = raw_per_case_metrics(memo_raw, run.window_len)
    comparison = compare_pair(run, dice_metrics, memo_metrics, test_df)

    if save_summary_csvs:
        dice_summary.to_csv(run_dir / "dice_summary.csv", index=False)
        memo_summary.to_csv(run_dir / "memo_summary.csv", index=False)
    if save_per_trace_results:
        dice_metrics.to_csv(run_dir / "dice_per_trace.csv", index=False)
        memo_metrics.to_csv(run_dir / "memo_per_trace.csv", index=False)
        comparison.to_csv(run_dir / "comparison.csv", index=False)
    if save_full_alignments:
        with (run_dir / "dice_raw_results.pkl").open("wb") as handle:
            pickle.dump(dice_raw, handle)
        with (run_dir / "memo_raw_results.pkl").open("wb") as handle:
            pickle.dump(memo_raw, handle)

    aggregate = aggregate_comparison(run, comparison, dice_wall, memo_wall)
    (run_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True))
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paired DICE vs DICE-M memoization experiments.")
    parser.add_argument("--config", type=Path, default=Path("configs/dice_memo_comparison.toml"))
    parser.add_argument("--output-dir", type=Path, help="Override output directory from config.")
    parser.add_argument("--list-runs", action="store_true", help="Print planned runs and exit.")
    parser.add_argument("--validate-paths", action="store_true", help="Validate selected dataset/model paths and exit.")
    parser.add_argument("--run-id", action="append", help="Run only a specific run_id. Can be provided multiple times.")
    parser.add_argument("--limit-runs", type=int, help="Run only the first N selected runs.")
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Rerun selected experiments even if aggregate.json and comparison.csv already exist.",
    )
    parser.add_argument(
        "--max-input-traces",
        type=int,
        help="Override n_traces after filtering. Useful for smoke tests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    runs = build_runs(config, max_input_traces=args.max_input_traces)

    if args.run_id:
        requested = set(args.run_id)
        runs = [run for run in runs if run.run_id in requested]
        missing = sorted(requested - {run.run_id for run in runs})
        if missing:
            print(f"Unknown run_id(s): {', '.join(missing)}", file=sys.stderr)
            return 2

    if args.limit_runs is not None:
        runs = runs[: args.limit_runs]

    for run in runs:
        split = run.split_name or "all"
        train_desc = (
            f"fixed_train={len(run.train_cases)}"
            if run.train_cases is not None
            else f"n_train={run.n_train_traces}"
        )
        print(
            f"{run.run_id}: dataset={run.dataset_name} category={run.category} "
            f"split={split} window={run.window_len} {train_desc} "
            f"unique_test={run.unique_test_variants} file={run.subfolder}/{run.df_name}"
        )

    if args.validate_paths:
        for run in runs:
            validate_run_paths(run)
        print(f"Validated paths for {len(runs)} run(s).")

    if args.list_runs or args.validate_paths:
        return 0

    if not runs:
        print("No runs selected.", file=sys.stderr)
        return 2

    defaults = config["defaults"]
    output_root = args.output_dir or Path(config["paths"]["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    aggregate_rows = []
    for index, run in enumerate(runs, start=1):
        print(f"\n[{index}/{len(runs)}] Running {run.run_id}")
        aggregate_rows.append(
            run_experiment(
                run,
                output_root=output_root,
                save_summary_csvs=bool(defaults.get("save_summary_csvs", True)),
                save_per_trace_results=bool(defaults.get("save_per_trace_results", True)),
                save_full_alignments=bool(defaults.get("save_full_alignments", False)),
                force_rerun=args.force_rerun,
            )
        )

    aggregate_df = pd.DataFrame(aggregate_rows)
    aggregate_path = output_root / "aggregate_comparison.csv"
    aggregate_df.to_csv(aggregate_path, index=False)
    print(f"\nWrote aggregate comparison: {aggregate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
