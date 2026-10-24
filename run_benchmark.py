# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
End-to-end benchmark of LLMs on Cloud AI 100 Ultra through QEfficient.

One command does the whole flow: cache variables, idle-card check, SDK and commit
record, the cases per model through benchmark.py, one CSV per model, one CSV with
every model (qaic_benchmark_all.csv, the file to share) and a Markdown report.

    python3 run_benchmark.py --models IFM/K2-Horizon-7B                     # default matrix
    python3 run_benchmark.py --models IFM/K2-Horizon-7B --only mx_4dev_ctx4k --repeats 5
    python3 run_benchmark.py --models IFM/K2-Horizon-7B --workload client --repeats 1
    python3 run_benchmark.py --models A B C --workload sweep                # context x prompt x output x users grid
    python3 run_benchmark.py --models A --workload sweep --devices 0-7      # two cards
    python3 run_benchmark.py --models A --workload sweep --resume           # skip the cases already measured
    python3 run_benchmark.py --models A --dry-run                           # model facts and plan only, no compile
    python3 run_benchmark.py --models A --report-only                       # rebuild the report and combined CSV

Any text causal LM QEfficient supports works. Model facts (layers, KV heads, head dim,
max context, parameter count) come from the Hub config and safetensors metadata;
models.json holds the per-model settings that differ from the flags (precision,
devices, context cap, parameter count, matrices).

Results land in <workspace>/qaic_benchmark[_<workload>]/ (workspace defaults to
/local/mnt/workspace/$USER). Run it from a venv with QEfficient installed.
"""

import argparse
import csv
import gc
import getpass
import importlib.util
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

from model_info import estimate_memory_gb, fits, load_model_info, memory_budget_gb

HERE = Path(__file__).resolve().parent
MATRICES = HERE / "matrices"
MODELS_FILE = HERE / "models.json"
QAIC_TOOLS = Path("/opt/qti-aic/tools")
# hand-written case files; "sweep" is built by sweep_cases, too many cases for a file.
# "serving": 1024-token prompts, long outputs, 1 to 32 users, the shapes in the customer's reference sheet
# "client": the four serving cases worth sharing (1, 16 and 32 users at 1k/1k, plus the 10k-prompt TTFT)
WORKLOADS = {
    "default": "benchmark_matrix.json",
    "serving": "benchmark_matrix_serving.json",
    "client": "benchmark_matrix_client.json",
    "sweep": None,
}
DEFAULT_DEVICES = [0, 1, 2, 3]  # one Ultra card
DEFAULT_PRECISION = "mxfp6"
# "sweep": every context x every prompt length that fits it x every output length x every user count;
# one compile per (context, users), the prompt and output lengths reuse it
SWEEP_CTX_LENS = [2048, 4096, 8192, 16384, 32768]
SWEEP_USERS = [1, 4, 8, 16, 32]
SWEEP_OUTPUT_LENS = [128, 1024]
SWEEP_PREFILL_SEQ_LEN = 1024
OUT_PREFIX = "qaic_benchmark"
COMBINED_CSV = f"{OUT_PREFIX}_all.csv"

REPORT_COLUMNS = [
    ("id", "Case"),
    ("num_devices", "Devices"),
    ("precision", "Precision"),
    ("ctx_len", "Context"),
    ("input_len", "Input"),
    ("generation_len", "Output"),
    ("batch_size", "Batch"),
    ("ttft_s", "TTFT s"),
    ("decode_tok_s", "Decode tok/s"),
    ("decode_tok_s_per_stream", "Per stream"),
    ("itl_ms", "ITL ms"),
    ("decode_board_w", "Board W"),
    ("tok_s_per_w", "tok/s per W"),
    ("joules_per_output_token", "J per token"),
    ("usd_per_1m_output_tokens", "$ per 1M out"),
    ("tok_s_per_dollar", "tok/s per $"),
]


def model_slug(model_name):
    return model_name.replace("/", "__")


def run(cmd, timeout=30):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def parse_qaic_status(text):
    """QID -> {"status": str, "active": int} from `qaic-util -q` output."""
    cards = {}
    qid = None
    for line in text.splitlines():
        m = re.match(r"\s*QID\s+(\d+)", line)
        if m:
            qid = int(m.group(1))
            cards[qid] = {"status": "", "active": 0}
            continue
        if qid is None:
            continue
        m = re.match(r"\s*Status\s*:\s*(\S+)", line)
        if m:
            cards[qid]["status"] = m.group(1)
        m = re.match(r"\s*Networks Active\s*:\s*(\d+)", line)
        if m:
            cards[qid]["active"] = int(m.group(1))
    return cards


def setup_environment(workspace):
    """Caches on the local disk, never on the NFS home (quota). Existing values win."""
    defaults = {
        "HF_HOME": workspace / "hf",
        "QEFF_HOME": workspace / "qeff_cache",
        "QEFF_LOG_PATH": workspace / "qeff_logs",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, str(value))
    for key in ("HF_HOME", "QEFF_HOME", "QEFF_LOG_PATH"):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
    return {key: os.environ[key] for key in defaults}


def check_cards(devices, dry_run):
    """Every device we will use must be Ready with no network loaded, or the numbers are off."""
    problems = []
    if not (QAIC_TOOLS / "qaic-util").exists():
        problems.append(f"{QAIC_TOOLS / 'qaic-util'} not found, is the Cloud AI SDK installed on this host?")
    else:
        cards = parse_qaic_status(run([str(QAIC_TOOLS / "qaic-util"), "-q"]))
        for d in sorted(devices):
            if d not in cards:
                problems.append(f"device {d} is not listed by qaic-util")
            elif cards[d]["status"].lower() != "ready":
                problems.append(f"device {d} status is {cards[d]['status'] or 'unknown'}, expected Ready")
            elif cards[d]["active"]:
                problems.append(f"device {d} has {cards[d]['active']} network(s) active, something else is using it")
    for p in problems:
        print(("warning: " if dry_run else "error: ") + p)
    return not problems or dry_run


def parse_devices(text):
    """ "0-3" or "0,1,2,3" or "0-3,8-11" -> [0, 1, 2, 3, ...]."""
    devices = []
    for part in text.split(","):
        first, _, last = part.partition("-")
        devices += list(range(int(first), int(last or first) + 1))
    return devices


def load_profiles(path):
    return json.loads(path.read_text()) if path.exists() else {}


def resolve_model(model_name, profiles, args):
    """Flags win over models.json, models.json wins over the defaults."""
    profile = profiles.get(model_name, {})
    try:
        info = load_model_info(model_name, profile.get("params_b"))
    except Exception as exc:  # no config: the context cap and fit check are skipped, the run still goes
        print(f"warning: {model_name}: could not read the config ({exc})")
        info = None
    return {
        "name": model_name,
        "precision": args.precision or profile.get("precision", DEFAULT_PRECISION),
        "kv_cache": args.kv_cache,
        "devices": parse_devices(args.devices) if args.devices else profile.get("devices", DEFAULT_DEVICES),
        "max_ctx_len": profile.get("max_ctx_len") or (info.max_position_embeddings if info else None),
        "matrices": profile.get("matrices", {}),
        "info": info,
    }


def sweep_cases(model, args):
    mxfp6 = model["precision"] == "mxfp6"
    tag = f"{'mx' if mxfp6 else 'fp16'}_{len(model['devices'])}dev"
    prefill = args.prefill_seq_len
    cases = []
    for ctx_len in args.ctx_lens:
        input_lens = sorted({prefill, ctx_len // 2, ctx_len - prefill} - {0})
        for users in args.users:
            for input_len in input_lens:
                for generation_len in args.output_lens:
                    if input_len + generation_len > ctx_len:
                        continue
                    cases.append(
                        {
                            "id": f"{tag}_ctx{ctx_len // 1024}k_in{input_len}_out{generation_len}_c{users}",
                            "devices": model["devices"],
                            "mxfp6": mxfp6,
                            "mxint8_kv_cache": model["kv_cache"] == "mxint8",
                            "prefill_seq_len": prefill,
                            "ctx_len": ctx_len,
                            "input_len": input_len,
                            "generation_len": generation_len,
                            "batch_size": users,
                        }
                    )
    return cases


def load_cases(model, args):
    if args.matrix:
        path = Path(args.matrix)
    elif args.workload == "sweep":
        path = None
    else:
        path = MATRICES / model["matrices"].get(args.workload, WORKLOADS[args.workload])
    cases = sweep_cases(model, args) if path is None else json.loads(path.read_text())
    source = "the sweep" if path is None else path.name
    if args.only:
        unknown = set(args.only) - {c["id"] for c in cases}
        if unknown:
            raise SystemExit(f"unknown case id(s) {sorted(unknown)} in {source}")
        cases = [c for c in cases if c["id"] in args.only]
    for c in cases:
        c.setdefault("devices", model["devices"])
        if c["prefill_seq_len"] > c["ctx_len"]:
            raise SystemExit(f"{c['id']}: prefill_seq_len is larger than ctx_len")
        if c["input_len"] + c["generation_len"] > c["ctx_len"]:
            raise SystemExit(f"{c['id']}: input_len + generation_len does not fit in ctx_len")
        if not c["devices"]:
            raise SystemExit(f"{c['id']}: empty device list")
    return source, cases


def skip_reason(model, case, args):
    """Why a case is not run, or None. Skipped cases still get a row so the shared CSV shows them."""
    if model["max_ctx_len"] and case["ctx_len"] > model["max_ctx_len"]:
        return f"ctx_len {case['ctx_len']} is above the model's limit {model['max_ctx_len']}"
    estimate = estimate_memory_gb(model["info"], case)
    if not args.no_fit_check and not fits(estimate, len(case["devices"])):
        budget = memory_budget_gb(len(case["devices"]))
        return f"estimated {estimate} GB (weights + KV at full context) exceeds {budget:.0f} GB"
    return None


def compile_key(case):
    """Cases with the same key share one QPC; only the prompt and output lengths differ."""
    return (
        tuple(case["devices"]),
        case.get("mxfp6", False),
        case.get("mxint8_kv_cache", False),
        case["prefill_seq_len"],
        case["ctx_len"],
        case.get("batch_size", 1),
    )


def git_info(path):
    commit = run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"]).strip()
    branch = run(["git", "-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"]).strip()
    return commit or "unknown", branch or "unknown"


def qefficient_checkout():
    """The source tree of the installed QEfficient (an editable install), found without importing it."""
    spec = importlib.util.find_spec("QEfficient")
    return Path(spec.origin).resolve().parents[1] if spec and spec.origin else None


def collect_metadata(env):
    qeff = qefficient_checkout()
    commit, branch = git_info(qeff) if qeff else ("not installed", "not installed")
    bench_commit, _ = git_info(HERE)
    sdk = run([str(QAIC_TOOLS / "qaic-version-util"), "--apps"]).strip()
    return {
        "host": socket.gethostname(),
        "user": getpass.getuser(),
        "python": platform.python_version(),
        "qefficient_commit": commit,
        "qefficient_branch": branch,
        "benchmark_commit": bench_commit,
        "qaic_sdk": sdk.splitlines()[0] if sdk else "unknown",
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        **env,
    }


def benchmark_model(model, cases, args, out_csv):
    """Runs every case through benchmark.py and appends the rows to out_csv, one compile per QPC shape."""
    # imported here so the cache variables are in the environment before QEfficient loads
    sys.path.insert(0, str(HERE))
    from transformers import AutoTokenizer

    import benchmark

    model_name = model["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    case_args = argparse.Namespace(
        model_name=model_name,
        repeats=args.repeats,
        num_cores=args.num_cores,
        card_price_usd=args.card_price_usd,
        card_hour_usd=args.card_hour_usd,
        electricity_usd_per_kwh=args.electricity_usd_per_kwh,
    )
    failures = []
    if out_csv.exists():
        with out_csv.open(newline="") as f:
            header = next(csv.reader(f), [])
        if header != benchmark.CSV_FIELDS:  # columns changed since that file was written, keep it aside
            out_csv.rename(out_csv.with_suffix(".old.csv"))
    done = set()
    if args.resume and out_csv.exists():
        done = {row["id"] for row in latest_rows(out_csv) if row.get("status", "ok") == "ok"}
    write_header = not out_csv.exists()

    def write(row):
        nonlocal write_header
        with out_csv.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=benchmark.CSV_FIELDS)
            if write_header:
                writer.writeheader()
                write_header = False
            writer.writerow(row)
        print(json.dumps(row, indent=1), flush=True)

    groups = {}
    for case in cases:
        if case["id"] in done:
            print(f"{model_name} / {case['id']}: already measured, skipped (--resume)")
            continue
        reason = skip_reason(model, case, args)
        if reason:
            estimate = estimate_memory_gb(model["info"], case)
            write({**benchmark.failed_row(case, model_name, "skipped", reason), "est_memory_gb": estimate})
            continue
        groups.setdefault(compile_key(case), []).append(case)

    for group in groups.values():
        print(f"\n=== {model_name} / compile for {', '.join(c['id'] for c in group)} ===", flush=True)
        try:
            qeff_model, qpc_path = benchmark.load_and_compile(group[0], case_args)
        except Exception as exc:  # one failed compile must not lose the other shapes
            print(f"compile failed: {exc}", flush=True)
            for case in group:
                failures.append({"id": case["id"], "error": f"compile: {exc}"})
                write(benchmark.failed_row(case, model_name, "failed", f"compile: {exc}"))
            continue
        for case in group:
            print(f"\n=== {model_name} / {case['id']} ===", flush=True)
            try:
                row = benchmark.measure(qeff_model, qpc_path, case, case_args, tokenizer)
            except Exception as exc:  # one failed case must not lose the others
                print(f"{case['id']} failed: {exc}", flush=True)
                failures.append({"id": case["id"], "error": str(exc)})
                row = benchmark.failed_row(case, model_name, "failed", str(exc))
            write({**row, "est_memory_gb": estimate_memory_gb(model["info"], case)})
        del qeff_model
        gc.collect()
    return failures


def precision_label(row):
    weights = "MXFP6" if row["mxfp6"] == "True" else "fp16"
    return f"{weights} + MXINT8 KV" if row["mxint8_kv_cache"] == "True" else weights


def latest_rows(csv_path):
    """One row per case id, the last one written wins (reruns append)."""
    rows = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows[row["id"]] = row
    return list(rows.values())


def write_combined_csv(out_dir, models):
    """Every model's latest rows in one file, the one to import and share."""
    rows, fields = [], []
    for model_name in models:
        csv_path = out_dir / f"{model_slug(model_name)}.csv"
        if not csv_path.exists():
            continue
        with csv_path.open(newline="") as f:
            header = next(csv.reader(f), [])
        fields += [field for field in header if field not in fields]
        rows += latest_rows(csv_path)
    combined = out_dir / COMBINED_CSV
    with combined.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return combined


def write_report(out_dir, models):
    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    lines = ["# LLM benchmark on Cloud AI 100 Ultra", ""]
    if meta:
        lines += [
            f"Host {meta.get('host')}, SDK {meta.get('qaic_sdk')}, QEfficient {meta.get('qefficient_commit')} "
            f"({meta.get('qefficient_branch')}), started {meta.get('started')}.",
            "",
        ]
    lines += [
        "Decode tok/s is the sum over all streams; per stream is what one user sees. Board W is the peak board "
        "power sampled during the timed runs (150 W TDP per card of 4 devices). tok/s per W = decode tok/s / board W. "
        "ITL is the mean time between two tokens of one stream.",
        "",
        f"$ per 1M out = card rental (${meta.get('card_hour_usd', '?')} per card-hour, a case pays for the share of "
        "the card it uses) / output tokens per hour, end to end "
        "with prefill included. J per token = mean board power / output tok/s, card only. Every column, including "
        f"$ per 1M input+output tokens and electricity cost, is in `{COMBINED_CSV}`.",
        "",
    ]
    columns = REPORT_COLUMNS
    for model_name in models:
        csv_path = out_dir / f"{model_slug(model_name)}.csv"
        lines += [f"## {model_name}", ""]
        info = meta.get("models", {}).get(model_name, {}).get("info")
        if info:
            lines += [
                f"{info['params_b']}B parameters, {info['num_layers']} layers, {info['num_kv_heads']} KV heads, "
                f"head_dim {info['head_dim']}.",
                "",
            ]
        if not csv_path.exists():
            lines += ["No results.", ""]
            continue
        all_rows = latest_rows(csv_path)
        rows = [r for r in all_rows if r.get("status", "ok") == "ok"]
        not_run = [r for r in all_rows if r.get("status", "ok") != "ok"]
        cols = [c for c in columns if any(r.get(c[0]) for r in rows) or c[0] == "precision"]
        lines.append("| " + " | ".join(label for _, label in cols) + " |")
        lines.append("|" + "---|" * len(cols))
        for row in rows:
            row = {**row, "precision": precision_label(row)}
            lines.append("| " + " | ".join(str(row.get(key, "")) for key, _ in cols) + " |")
        lines.append("")
        if not_run:
            lines += ["Not measured:", ""] + [f"- `{r['id']}` ({r['status']}): {r['error']}" for r in not_run] + [""]
        lines += [
            f"Settings for every row: `prefill_seq_len` and `ctx_len` per case in the matrix, `num_cores` "
            f"{meta.get('num_cores', 16)}, repeats {meta.get('repeats', 3)} after one warm-up, "
            f"`mos=1`, `aic_enable_depth_first`, ONNX subfunctions. Full columns in `{csv_path.name}`.",
            "",
        ]
    report = out_dir / "REPORT.md"
    report.write_text("\n".join(lines))
    return report


def main():
    parser = argparse.ArgumentParser(description="LLM benchmark on Cloud AI 100 Ultra through QEfficient")
    parser.add_argument("--models", nargs="+", required=True, help="Hugging Face model ids")
    parser.add_argument("--workload", choices=sorted(WORKLOADS), default="default", help="which case set to run")
    parser.add_argument("--matrix", help="case file for every model (overrides --workload)")
    parser.add_argument("--only", nargs="*", help="run only these case ids")
    parser.add_argument("--models-file", default=str(MODELS_FILE), help="per-model settings")
    parser.add_argument("--devices", help='devices for the sweep and for cases without "devices", e.g. 0-3 or 0-7')
    parser.add_argument(
        "--precision", choices=["mxfp6", "fp16"], help="sweep weights (default: models.json, else mxfp6)"
    )
    parser.add_argument("--kv-cache", choices=["mxint8", "fp16"], default="mxint8", help="sweep KV cache precision")
    parser.add_argument("--ctx-lens", nargs="+", type=int, default=SWEEP_CTX_LENS, help="sweep context lengths")
    parser.add_argument("--users", nargs="+", type=int, default=SWEEP_USERS, help="sweep concurrent users")
    parser.add_argument("--output-lens", nargs="+", type=int, default=SWEEP_OUTPUT_LENS, help="sweep output lengths")
    parser.add_argument("--prefill-seq-len", type=int, default=SWEEP_PREFILL_SEQ_LEN, help="sweep prefill chunk")
    parser.add_argument(
        "--repeats", type=int, help="timed runs per case after one warm-up (default 3, 1 for the sweep)"
    )
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument("--card-price-usd", type=float, help="card price, fills tok/s per dollar")
    parser.add_argument(
        "--card-hour-usd",
        type=float,
        default=0.80,
        help="rental $ per card-hour for the cost columns (Cirrascale 8x Ultra server $4,699/month = $0.80 per card)",
    )
    parser.add_argument("--electricity-usd-per-kwh", type=float, default=0.13, help="for the energy cost column")
    parser.add_argument("--resume", action="store_true", help="skip the cases already measured in the CSVs")
    parser.add_argument("--no-fit-check", action="store_true", help="also try cases estimated not to fit")
    parser.add_argument(
        "--workspace", default=f"/local/mnt/workspace/{getpass.getuser()}", help="local disk for caches and results"
    )
    parser.add_argument("--out-dir", help=f"results folder (default: <workspace>/{OUT_PREFIX}[_<workload>])")
    parser.add_argument("--dry-run", action="store_true", help="model facts and plan only, no compile or run")
    parser.add_argument("--report-only", action="store_true", help="rebuild REPORT.md and the combined CSV")
    args = parser.parse_args()
    if args.repeats is None:
        args.repeats = 1 if args.workload == "sweep" else 3

    workspace = Path(args.workspace)
    suffix = "" if args.workload == "default" else f"_{args.workload}"
    out_dir = Path(args.out_dir) if args.out_dir else workspace / f"{OUT_PREFIX}{suffix}"
    if args.report_only:
        print(write_combined_csv(out_dir, args.models))
        print(write_report(out_dir, args.models))
        return

    # before anything touches the Hub, so configs land in the workspace cache and not the NFS home
    env = {}
    try:
        env = setup_environment(workspace)
    except PermissionError as exc:
        if not args.dry_run:
            raise SystemExit(f"{exc}: pass --workspace with a writable local path")
        print(f"warning: {exc}, using the default caches for the dry run")

    profiles = load_profiles(Path(args.models_file))
    plan = []
    for model_name in args.models:
        model = resolve_model(model_name, profiles, args)
        source, cases = load_cases(model, args)
        plan.append((model, source, cases))
    devices = {d for _, _, cases in plan for c in cases for d in c["devices"]}

    print(f"workspace {workspace}, results in {out_dir}")
    for model, source, cases in plan:
        info = model["info"]
        facts = (
            f"{info.params_b}B params, {info.num_layers} layers, {info.num_kv_heads} KV heads, head_dim {info.head_dim}"
            if info
            else "no config"
        )
        skipped = {c["id"]: skip_reason(model, c, args) for c in cases}
        skipped = {k: v for k, v in skipped.items() if v}
        compiles = {compile_key(c) for c in cases if c["id"] not in skipped}
        print(
            f"{model['name']}: {facts}; {model['precision']} weights, max ctx {model['max_ctx_len']}, "
            f"devices {model['devices']}"
        )
        print(f"  {len(cases)} cases from {source}, {len(compiles)} compiles, {len(skipped)} skipped")
        if len(cases) <= 12:
            print("  " + ", ".join(c["id"] for c in cases))
        for case_id, reason in skipped.items():
            print(f"  skip {case_id}: {reason}")
    if not check_cards(devices, args.dry_run):
        raise SystemExit(1)
    if "VIRTUAL_ENV" not in os.environ:
        print("warning: no venv active, QEfficient may not import")
    if args.dry_run:
        print("dry run, nothing compiled or run")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = collect_metadata(env)
    meta.update(
        {
            "workload": args.workload,
            "num_cores": args.num_cores,
            "repeats": args.repeats,
            "card_price_usd": args.card_price_usd,
            "card_hour_usd": args.card_hour_usd,
            "electricity_usd_per_kwh": args.electricity_usd_per_kwh,
            "models": {
                m["name"]: {
                    "precision": m["precision"],
                    "devices": m["devices"],
                    "max_ctx_len": m["max_ctx_len"],
                    "info": m["info"].as_dict() if m["info"] else None,
                }
                for m, _, _ in plan
            },
        }
    )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1), flush=True)

    failures = {}
    for model, _, cases in plan:
        out_csv = out_dir / f"{model_slug(model['name'])}.csv"
        failures[model["name"]] = benchmark_model(model, cases, args, out_csv)
        write_combined_csv(out_dir, args.models)  # up to date after every model, in case the run is stopped

    report = write_report(out_dir, args.models)
    print(f"\nreport: {report}")
    print(f"all models: {out_dir / COMBINED_CSV}")
    for model_name, failed in failures.items():
        for f in failed:
            print(f"failed: {model_name} / {f['id']}: {f['error']}")
    if any(failures.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
