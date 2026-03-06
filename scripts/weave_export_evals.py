#!/usr/bin/env python3
"""
Export W&B Weave evaluation data via the v2 Evaluation REST API.

Uses the dedicated v2 evaluation endpoints rather than the general-purpose
calls/stream_query endpoint. These endpoints surface first-class evaluation
concepts: evaluation runs, predictions, scores, and scorers.

API reference: https://trace.wandb.ai/docs

Key endpoints used:
    GET  /v2/{entity}/{project}/evaluation_runs
    GET  /v2/{entity}/{project}/evaluation_runs/{evaluation_run_id}
    GET  /v2/{entity}/{project}/predictions/{prediction_id}
    GET  /v2/{entity}/{project}/scorers/{object_id}/versions/{digest}
    POST /v2/{entity}/{project}/eval_results/query

Usage:
    # List recent evaluation runs
    python weave_export_evals.py --entity my-team --project my-project

    # Export a specific evaluation run to JSON (stdout)
    python weave_export_evals.py --entity my-team --project my-project --eval-run-id <id>

    # Export to a JSON file
    python weave_export_evals.py --entity my-team --project my-project --eval-run-id <id> -o results.json

    # Export to CSV
    python weave_export_evals.py --entity my-team --project my-project --eval-run-id <id> --format csv

Requirements:
    pip install requests

Authentication:
    Set the WANDB_API_KEY environment variable with your W&B API key.
    Get your key at: https://wandb.ai/settings
"""

import argparse
import csv
import json
import os
import sys

import requests

TRACE_API_BASE = "https://trace.wandb.ai"


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def get_api_key():
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        print("Error: Set the WANDB_API_KEY environment variable.", file=sys.stderr)
        print("Get your key at: https://wandb.ai/settings", file=sys.stderr)
        sys.exit(1)
    return key


def api_get(path, api_key, params=None):
    resp = requests.get(
        f"{TRACE_API_BASE}{path}",
        params=params,
        auth=("api", api_key),
    )
    resp.raise_for_status()
    return resp


def api_post(path, payload, api_key):
    resp = requests.post(
        f"{TRACE_API_BASE}{path}",
        json=payload,
        auth=("api", api_key),
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    return resp


def parse_jsonl(resp):
    results = []
    for line in resp.text.strip().split("\n"):
        line = line.strip()
        if line:
            results.append(json.loads(line))
    return results


def ref_display_name(ref):
    if not ref or not isinstance(ref, str):
        return str(ref) if ref else ""
    if ref.startswith("weave:///"):
        parts = ref.split("/")
        if len(parts) >= 5:
            return parts[-1]
    return ref


def parse_ref_parts(ref):
    """Extract (object_id, digest) from a weave:/// ref.

    Example: weave:///entity/project/object/name:digest -> (name, digest)
    """
    if not ref or not ref.startswith("weave:///"):
        return None, None
    parts = ref.split("/")
    if len(parts) < 5:
        return None, None
    name_and_digest = parts[-1]
    if ":" in name_and_digest:
        object_id, digest = name_and_digest.split(":", 1)
        return object_id, digest
    return name_and_digest, None


# ---------------------------------------------------------------------------
# v2 API: Evaluation runs
# ---------------------------------------------------------------------------

def list_evaluation_runs(entity, project, api_key, limit=20):
    """GET /v2/{entity}/{project}/evaluation_runs"""
    resp = api_get(
        f"/v2/{entity}/{project}/evaluation_runs",
        api_key,
        params={"limit": limit},
    )
    return parse_jsonl(resp)


def read_evaluation_run(entity, project, eval_run_id, api_key):
    """GET /v2/{entity}/{project}/evaluation_runs/{evaluation_run_id}"""
    resp = api_get(
        f"/v2/{entity}/{project}/evaluation_runs/{eval_run_id}",
        api_key,
    )
    return resp.json()


# ---------------------------------------------------------------------------
# v2 API: Predictions
# ---------------------------------------------------------------------------

def read_prediction(entity, project, prediction_id, api_key):
    """GET /v2/{entity}/{project}/predictions/{prediction_id}"""
    resp = api_get(
        f"/v2/{entity}/{project}/predictions/{prediction_id}",
        api_key,
    )
    return resp.json()


# ---------------------------------------------------------------------------
# v2 API: Scorers
# ---------------------------------------------------------------------------

def read_scorer(entity, project, object_id, digest, api_key):
    """GET /v2/{entity}/{project}/scorers/{object_id}/versions/{digest}"""
    resp = api_get(
        f"/v2/{entity}/{project}/scorers/{object_id}/versions/{digest}",
        api_key,
    )
    return resp.json()


# ---------------------------------------------------------------------------
# v2 API: Eval results query (bridges old and new evaluations)
# ---------------------------------------------------------------------------

def query_eval_results(entity, project, eval_run_ids, api_key,
                       include_rows=True, include_summary=True,
                       include_raw_data=True, resolve_refs=True,
                       limit=None):
    """POST /v2/{entity}/{project}/eval_results/query

    Accepts both v2 evaluation_run_ids and legacy evaluation_call_ids.
    """
    body = {
        "evaluation_run_ids": eval_run_ids,
        "include_rows": include_rows,
        "include_summary": include_summary,
        "include_raw_data_rows": include_raw_data,
        "resolve_row_refs": resolve_refs,
    }
    if limit is not None:
        body["limit"] = limit

    resp = api_post(
        f"/v2/{entity}/{project}/eval_results/query",
        body,
        api_key,
    )
    return resp.json()


# ---------------------------------------------------------------------------
# Listing: v2 runs with fallback to legacy call IDs
# ---------------------------------------------------------------------------

def list_evaluations_via_calls(entity, project, api_key, limit=20):
    """Fallback: find Evaluation.evaluate calls via stream_query."""
    payload = {
        "project_id": f"{entity}/{project}",
        "query": {
            "$expr": {
                "$contains": {
                    "input": {"$getField": "op_name"},
                    "substr": {"$literal": "Evaluation.evaluate"},
                    "case_insensitive": False,
                }
            }
        },
        "sort_by": [{"field": "started_at", "direction": "desc"}],
        "limit": limit,
    }
    resp = api_post("/calls/stream_query", payload, api_key)
    calls = parse_jsonl(resp)

    results = []
    for call in calls:
        inputs = call.get("inputs", {})
        model_ref = inputs.get("model", inputs.get("self", ""))
        if isinstance(model_ref, dict):
            model_ref = model_ref.get("_ref", str(model_ref))

        results.append({
            "evaluation_run_id": call["id"],
            "evaluation": "",
            "model": str(model_ref),
            "status": call.get("summary", {}).get("weave", {}).get("status"),
            "started_at": call.get("started_at"),
            "finished_at": call.get("ended_at"),
            "summary": call.get("summary"),
            "_display_name": call.get("summary", {}).get("weave", {}).get("display_name"),
        })
    return results


def discover_evaluations(entity, project, api_key, limit=20):
    """List evaluation runs, trying v2 API first with legacy fallback."""
    runs = list_evaluation_runs(entity, project, api_key, limit=limit)
    if runs:
        return runs, "v2"
    runs = list_evaluations_via_calls(entity, project, api_key, limit=limit)
    return runs, "legacy"


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def build_export(entity, project, eval_run_id, api_key):
    """Assemble full export data for an evaluation run."""

    # 1. Read evaluation run metadata
    print(f"Reading evaluation run {eval_run_id}...", file=sys.stderr)
    try:
        eval_run = read_evaluation_run(entity, project, eval_run_id, api_key)
    except requests.HTTPError:
        eval_run = {
            "evaluation_run_id": eval_run_id,
            "evaluation": "",
            "model": "",
        }

    # 2. Query eval results (works for both v2 and legacy evaluations)
    print("Querying evaluation results...", file=sys.stderr)
    results = query_eval_results(
        entity, project, [eval_run_id], api_key,
        include_rows=True,
        include_summary=True,
        include_raw_data=True,
        resolve_refs=True,
    )

    summary_data = results.get("summary")
    rows = results.get("rows", [])
    total_rows = results.get("total_rows", len(rows))
    print(f"Found {total_rows} row(s) across {len(rows)} returned.", file=sys.stderr)

    # Enrich eval_run with summary info if available
    if summary_data and summary_data.get("evaluations"):
        eval_summary = summary_data["evaluations"][0]
        if not eval_run.get("model"):
            eval_run["model"] = eval_summary.get("model_ref", "")
        if not eval_run.get("evaluation"):
            eval_run["evaluation"] = eval_summary.get("evaluation_ref", "")
        eval_run["_display_name"] = eval_summary.get("display_name")

    # 3. Try to read individual predictions via v2 endpoint
    #    (available when evaluations are created through the v2 API)
    prediction_details = {}
    for row in rows:
        for ev in row.get("evaluations", []):
            for trial in ev.get("trials", []):
                pred_call_id = trial.get("predict_call_id")
                if pred_call_id:
                    try:
                        pred = read_prediction(entity, project, pred_call_id, api_key)
                        prediction_details[pred_call_id] = pred
                    except requests.HTTPError:
                        pass

    # 4. Look up scorer definitions via v2 scorers endpoint
    scorer_cache = {}
    if summary_data:
        for ev_summary in summary_data.get("evaluations", []):
            for stat in ev_summary.get("scorer_stats", []):
                scorer_key = stat.get("scorer_key", "")
                if scorer_key not in scorer_cache:
                    object_id, digest = parse_ref_parts(scorer_key)
                    if object_id and digest:
                        try:
                            scorer_info = read_scorer(
                                entity, project, object_id, digest, api_key)
                            scorer_cache[scorer_key] = scorer_info
                        except requests.HTTPError:
                            pass

    # 5. Assemble per-prediction rows
    predictions = []
    for row in rows:
        dataset_row = row.get("raw_data_row")
        row_digest = row.get("row_digest", "")

        for ev in row.get("evaluations", []):
            for trial in ev.get("trials", []):
                call_id = trial.get("predict_and_score_call_id", "")

                pred_detail = prediction_details.get(
                    trial.get("predict_call_id"), {})

                predictions.append({
                    "predict_and_score_call_id": call_id,
                    "predict_call_id": trial.get("predict_call_id"),
                    "row_digest": row_digest,
                    "inputs": pred_detail.get("inputs", dataset_row),
                    "output": trial.get("model_output"),
                    "scores": trial.get("scores", {}),
                    "model_latency_seconds": trial.get("model_latency_seconds"),
                    "total_tokens": trial.get("total_tokens"),
                })

    # 6. Assemble scorer stats summary
    scorer_stats = []
    if summary_data:
        for ev_summary in summary_data.get("evaluations", []):
            for stat in ev_summary.get("scorer_stats", []):
                key = stat["scorer_key"]
                path = stat.get("path")
                full_key = f"{key}.{path}" if path else key

                entry = {
                    "scorer": full_key,
                    "value_type": stat.get("value_type"),
                    "trial_count": stat.get("trial_count", 0),
                }
                if stat.get("value_type") == "binary":
                    entry["pass_rate"] = stat.get("pass_rate")
                    entry["pass_true_count"] = stat.get("pass_true_count", 0)
                    entry["pass_known_count"] = stat.get("pass_known_count", 0)
                elif stat.get("value_type") == "continuous":
                    entry["numeric_mean"] = stat.get("numeric_mean")
                    entry["numeric_count"] = stat.get("numeric_count", 0)

                scorer_info = scorer_cache.get(key, {})
                if scorer_info:
                    entry["scorer_name"] = scorer_info.get("name")
                    entry["scorer_description"] = scorer_info.get("description")

                scorer_stats.append(entry)

    return {
        "evaluation_run": {
            "evaluation_run_id": eval_run.get("evaluation_run_id"),
            "evaluation_ref": eval_run.get("evaluation"),
            "evaluation_name": ref_display_name(eval_run.get("evaluation", "")),
            "model_ref": eval_run.get("model"),
            "model_name": ref_display_name(eval_run.get("model", "")),
            "display_name": eval_run.get("_display_name"),
            "status": eval_run.get("status"),
            "started_at": eval_run.get("started_at"),
            "finished_at": eval_run.get("finished_at"),
            "total_rows": total_rows,
        },
        "scorer_stats": scorer_stats,
        "predictions": predictions,
    }


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def format_summary(summary):
    if not summary:
        return "    (none)"
    lines = []
    for key, value in summary.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                lines.append(f"    {key}.{sub_key}: {sub_value}")
        elif isinstance(value, float):
            lines.append(f"    {key}: {value:.4f}")
        else:
            lines.append(f"    {key}: {value}")
    return "\n".join(lines) if lines else "    (none)"


def format_scorer_stats(scorer_stats):
    if not scorer_stats:
        return "    (none)"
    lines = []
    for stat in scorer_stats:
        name = stat["scorer"]
        if stat.get("value_type") == "binary":
            rate = stat.get("pass_rate")
            rate_str = f"{rate:.1%}" if rate is not None else "N/A"
            lines.append(
                f"    {name}: {rate_str} pass rate "
                f"({stat.get('pass_true_count', 0)}/{stat.get('pass_known_count', 0)})"
            )
        elif stat.get("value_type") == "continuous":
            mean = stat.get("numeric_mean")
            mean_str = f"{mean:.4f}" if mean is not None else "N/A"
            lines.append(f"    {name}: mean={mean_str} (n={stat.get('numeric_count', 0)})")
        else:
            lines.append(f"    {name}: {stat}")
    return "\n".join(lines)


def export_json(data, output_file=None):
    text = json.dumps(data, indent=2, default=str)
    if output_file:
        with open(output_file, "w") as f:
            f.write(text)
        print(f"Exported {len(data['predictions'])} predictions to {output_file}",
              file=sys.stderr)
    else:
        print(text)


def export_csv(data, output_file):
    predictions = data.get("predictions", [])
    if not predictions:
        print("No predictions to export.", file=sys.stderr)
        return

    all_score_paths = set()
    for p in predictions:
        scores = p.get("scores", {})
        for scorer_key, value in scores.items():
            if isinstance(value, dict):
                for sub_key in value:
                    all_score_paths.add(f"{scorer_key}.{sub_key}")
            else:
                all_score_paths.add(scorer_key)
    score_cols = sorted(all_score_paths)

    fieldnames = (
        ["predict_and_score_call_id", "row_digest", "inputs", "output"]
        + [f"score.{c}" for c in score_cols]
    )

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for p in predictions:
            row = {
                "predict_and_score_call_id": p.get("predict_and_score_call_id", ""),
                "row_digest": p.get("row_digest", ""),
                "inputs": json.dumps(p.get("inputs", {}), default=str),
                "output": json.dumps(p.get("output", {}), default=str),
            }

            scores = p.get("scores", {})
            for col in score_cols:
                parts = col.split(".", 1)
                if len(parts) == 2 and isinstance(scores.get(parts[0]), dict):
                    row[f"score.{col}"] = scores[parts[0]].get(parts[1], "")
                else:
                    row[f"score.{col}"] = scores.get(col, "")
            writer.writerow(row)

    print(f"Exported {len(predictions)} predictions to {output_file}",
          file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export W&B Weave evaluation data via the v2 Evaluation REST API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # List recent evaluation runs
  python weave_export_evals.py --entity my-team --project my-project

  # Export a specific evaluation run to JSON (by UUID or list index)
  python weave_export_evals.py --entity my-team --project my-project --eval-run-id <id>
  python weave_export_evals.py --entity my-team --project my-project --eval-run-id 0

  # Export to CSV
  python weave_export_evals.py --entity my-team --project my-project --eval-run-id <id> --format csv -o results.csv
""",
    )
    parser.add_argument("--entity", required=True,
                        help="W&B entity (team or username)")
    parser.add_argument("--project", required=True,
                        help="W&B project name")
    parser.add_argument("--eval-run-id",
                        help="Evaluation run ID or list index (e.g. 0, 1) to export (omit to list runs)")
    parser.add_argument("--format", choices=["json", "csv"], default="json",
                        help="Output format (default: json)")
    parser.add_argument("-o", "--output",
                        help="Output file path (default: stdout for JSON)")
    parser.add_argument("--limit", type=int, default=20,
                        help="Max evaluation runs to list (default: 20)")

    args = parser.parse_args()
    api_key = get_api_key()

    # -- List mode --
    if not args.eval_run_id:
        print(f"Fetching evaluation runs from {args.entity}/{args.project}...\n",
              file=sys.stderr)
        runs, source = discover_evaluations(
            args.entity, args.project, api_key, limit=args.limit)

        if not runs:
            print("No evaluation runs found.", file=sys.stderr)
            return

        print(f"Found {len(runs)} evaluation(s) (via {source} API):\n")
        for i, run in enumerate(runs):
            display = run.get("_display_name") or ref_display_name(run.get("model", ""))
            print(f"[{i}] {run['evaluation_run_id']}")
            print(f"    Name:       {display}")
            print(f"    Status:     {run.get('status', '')}")
            print(f"    Started:    {run.get('started_at', '')}")
            print(f"    Model:      {ref_display_name(run.get('model', ''))}")
            print(f"    Evaluation: {ref_display_name(run.get('evaluation', ''))}")
            if run.get("summary"):
                print(f"    Summary:")
                print(format_summary(run["summary"]))
            print()

        print("Re-run with --eval-run-id <id or index> to export a specific evaluation.")
        return

    # -- Export mode --
    eval_run_id = args.eval_run_id

    # Support index-based selection (e.g. --eval-run-id 0)
    if eval_run_id.isdigit():
        idx = int(eval_run_id)
        print(f"Resolving index [{idx}] to evaluation run ID...", file=sys.stderr)
        runs, source = discover_evaluations(
            args.entity, args.project, api_key, limit=max(args.limit, idx + 1))
        if idx >= len(runs):
            print(f"Error: Index {idx} out of range. Only {len(runs)} evaluation(s) found.",
                  file=sys.stderr)
            sys.exit(1)
        eval_run_id = runs[idx]["evaluation_run_id"]
        display = runs[idx].get("_display_name") or ref_display_name(runs[idx].get("model", ""))
        print(f"Selected [{idx}] {eval_run_id} ({display})\n", file=sys.stderr)

    data = build_export(args.entity, args.project, eval_run_id, api_key)

    # Show scorer stats summary on stderr
    if data.get("scorer_stats"):
        print("\nScorer stats:", file=sys.stderr)
        print(format_scorer_stats(data["scorer_stats"]), file=sys.stderr)
        print(file=sys.stderr)

    if args.format == "csv":
        output_file = args.output or f"eval_{args.eval_run_id[:8]}.csv"
        export_csv(data, output_file)
    else:
        export_json(data, args.output)


if __name__ == "__main__":
    main()
