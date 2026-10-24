# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Benchmark a K2 Horizon model on Cloud AI 100 over a matrix of configurations.

For every entry in the matrix file: compile (cached by QEfficient), run a warm-up,
then `--repeats` timed runs while sampling board power with qaic-util, and append
one CSV row with TTFT, decode tokens/s, inter-token latency, total tokens/s,
board power, tokens/s per watt and the cost of a million tokens.

Cost is rental time: `--card-hour-usd` (one Ultra card, 4 devices) over the tokens
the card produces in an hour, end to end (prefill included). Energy is the mean
board power over the timed runs, card only (no host, no cooling).

    python3 benchmark.py --matrix matrices/benchmark_matrix.json --out results.csv
    python3 benchmark.py --matrix matrices/benchmark_matrix.json --only mx_4dev_ctx4k --repeats 5
"""

import argparse
import csv
import json
import re
import statistics
import subprocess
import threading
import time
from pathlib import Path

from QEfficient import QEFFAutoModelForCausalLM
from transformers import AutoTokenizer

QAIC_UTIL = "/opt/qti-aic/tools/qaic-util"
DEVICES_PER_CARD = 4  # one AI 100 Ultra card has four SoCs
# Cirrascale rents an 8x AI 100 Ultra server at $4,699 a month: 4699 / 8 / 730 h
DEFAULT_CARD_HOUR_USD = 0.80
DEFAULT_ELECTRICITY_USD_PER_KWH = 0.13  # about the US commercial average
CSV_FIELDS = [
    "id",
    "model",
    "status",
    "error",
    "devices",
    "num_devices",
    "mxfp6",
    "mxint8_kv_cache",
    "prefill_seq_len",
    "ctx_len",
    "input_len",
    "generation_len",
    "batch_size",
    "repeats",
    "ttft_s",
    "prefill_tok_s",
    "decode_tok_s",
    "total_tok_s",
    "e2e_s",
    "decode_tok_s_per_stream",
    "itl_ms",
    "idle_board_w",
    "decode_board_w",
    "mean_board_w",
    "tok_s_per_w",
    "joules_per_output_token",
    "card_hour_usd",
    "usd_per_1m_output_tokens",
    "usd_per_1m_tokens",
    "electricity_usd_per_kwh",
    "energy_usd_per_1m_output_tokens",
    "tok_s_per_dollar",
    "est_memory_gb",
    "qpc",
    "timestamp",
]


def parse_board_power(text):
    """Map QID -> Board power(Watts) from `qaic-util -q` output."""
    powers = {}
    qid = None
    for line in text.splitlines():
        m = re.match(r"\s*QID\s+(\d+)", line)
        if m:
            qid = int(m.group(1))
            continue
        m = re.match(r"\s*Board power\(Watts\):\s*([\d.]+)", line)
        if m and qid is not None:
            powers[qid] = float(m.group(1))
    return powers


def read_board_power(devices):
    try:
        out = subprocess.run([QAIC_UTIL, "-q"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    powers = parse_board_power(out)
    values = [powers[d] for d in devices if d in powers]
    # every device on one card reports the whole board, so take the max rather than the sum
    return max(values) if values else None


class PowerSampler:
    """Samples board power on a thread until stopped."""

    def __init__(self, devices, interval_s=0.5):
        self.devices = devices
        self.interval_s = interval_s
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            value = read_board_power(self.devices)
            if value is not None:
                self.samples.append(value)
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=15)

    def peak(self):
        return max(self.samples) if self.samples else None

    def mean(self):
        return round(statistics.mean(self.samples), 1) if self.samples else None


PROMPT_TEXT = """
The history of the printing press begins in the middle of the fifteenth century, when a goldsmith in Mainz
combined movable metal type, an oil-based ink and a modified wine press into a single workflow. Within fifty
years presses were running in more than two hundred European cities, and the cost of a book fell to a fraction
of what a hand-copied manuscript had demanded. Scholars could now compare editions, printers competed on
accuracy, and errors that had once propagated silently from copy to copy became visible and correctable.

A modern transformer language model processes text as a sequence of integer tokens. During the prefill phase
the whole prompt is pushed through the network at once, and the attention layers store a key and a value tensor
for every token in every layer. During decode the model produces one token at a time, reading the stored cache
instead of recomputing earlier positions. Prefill is bound by compute, decode is bound by memory bandwidth, and
the balance between the two decides which hardware is the better fit for a given workload.

Coffee cultivation spread from the Ethiopian highlands to Yemen, where the beans were first roasted and brewed in
the form we recognise today. Coffee houses opened in Istanbul, then in Venice, London and Vienna, and became
places where merchants traded news, insurers priced ships and pamphleteers argued about politics. The drink was
banned more than once by rulers who distrusted the conversations it encouraged, and each ban failed.

The orbit of a satellite is set by its altitude and inclination. A low orbit at a few hundred kilometres completes
a lap in about ninety minutes and sees a narrow strip of the surface on each pass, which suits imaging and
weather observation. A geostationary orbit at roughly thirty-six thousand kilometres matches the rotation of the
planet, so the satellite appears fixed in the sky and a single antenna on the ground can stay pointed at it
without tracking. Between the two sit the navigation constellations, whose clocks must be corrected for both
special and general relativity or their position fixes would drift by kilometres within a day.

Sourdough bread rises because wild yeast and lactic acid bacteria ferment the sugars in flour. The yeast produces
carbon dioxide that is trapped by the gluten network, while the bacteria produce acids that give the crumb its
flavour and slow the growth of moulds. A baker keeps the culture alive by discarding part of it and feeding the
rest with fresh flour and water, and the ratio of flour to water to starter, together with the temperature of the
kitchen, controls how fast the dough proves.

Compilers for accelerator hardware translate a dataflow graph into a schedule of kernels, buffers and transfers.
The scheduler decides which tensors stay on chip, which spill to external memory and how operations are fused so
that intermediate results never leave the cores. A graph that fits in on-chip memory runs at the speed of the
arithmetic units; one that does not is paced by the memory interface. Quantisation shrinks the weights and the
cache so that more of the working set fits, and the trade is a small loss of precision for a large gain in
throughput.

The Silk Road was never a single road but a web of routes linking Chang'an with the Mediterranean through the
oases of Central Asia. Silk went west, but so did paper, gunpowder and the compass; glass, horses, silver and
religions went east. Caravans rarely travelled the full distance. Goods changed hands many times, and each
intermediary added a margin, so by the time a bolt of silk reached Rome it was worth its weight in gold.
"""


def build_prompt(tokenizer, input_len):
    """A prose prompt that tokenizes to exactly input_len tokens including the BOS token."""
    body = " ".join(PROMPT_TEXT.split())
    ids = tokenizer(body, add_special_tokens=False)["input_ids"]
    ids = ids * (input_len // len(ids) + 1)
    take = input_len - 1
    prompt = ""
    for _ in range(8):
        prompt = tokenizer.decode(ids[:take])
        got = len(tokenizer(prompt)["input_ids"])
        if got == input_len:
            break
        take += input_len - got
    return prompt


def case_fields(case, model_name):
    """The configuration columns of a row, shared by measured and failed rows."""
    devices = case["devices"]
    return {
        "id": case["id"],
        "model": model_name,
        "devices": " ".join(str(d) for d in devices),
        "num_devices": len(devices),
        "mxfp6": case.get("mxfp6", False),
        "mxint8_kv_cache": case.get("mxint8_kv_cache", False),
        "prefill_seq_len": case["prefill_seq_len"],
        "ctx_len": case["ctx_len"],
        "input_len": case["input_len"],
        "generation_len": case["generation_len"],
        "batch_size": case.get("batch_size", 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def failed_row(case, model_name, status, error):
    return {**case_fields(case, model_name), "status": status, "error": error}


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


def load_and_compile(case, args):
    batch_size = case.get("batch_size", 1)
    continuous_batching = batch_size > 1
    model = QEFFAutoModelForCausalLM.from_pretrained(
        args.model_name, trust_remote_code=True, continuous_batching=continuous_batching
    )
    compile_kwargs = dict(
        prefill_seq_len=case["prefill_seq_len"],
        ctx_len=case["ctx_len"],
        num_cores=args.num_cores,
        num_devices=len(case["devices"]),
        mxfp6_matmul=case.get("mxfp6", False),
        mxint8_kv_cache=case.get("mxint8_kv_cache", False),
        use_onnx_subfunctions=True,
        aic_enable_depth_first=True,
        mos=1,
    )
    if continuous_batching:
        compile_kwargs["full_batch_size"] = batch_size
    qpc_path = model.compile(**compile_kwargs)
    return model, qpc_path


def cost_fields(row, args):
    """Cost and energy per token from the measured throughput and power."""
    card_hour_usd = getattr(args, "card_hour_usd", DEFAULT_CARD_HOUR_USD)
    kwh_usd = getattr(args, "electricity_usd_per_kwh", DEFAULT_ELECTRICITY_USD_PER_KWH)
    out_tok_s = row["total_tok_s"]
    all_tok_s = out_tok_s + row["batch_size"] * row["input_len"] / row["e2e_s"]
    hour_usd = card_hour_usd * row["num_devices"] / DEVICES_PER_CARD
    fields = {
        "card_hour_usd": card_hour_usd,
        "usd_per_1m_output_tokens": round(hour_usd / (out_tok_s * 3600) * 1e6, 4),
        "usd_per_1m_tokens": round(hour_usd / (all_tok_s * 3600) * 1e6, 4),
        "electricity_usd_per_kwh": kwh_usd,
    }
    if row["mean_board_w"]:
        joules = row["mean_board_w"] / out_tok_s
        fields["joules_per_output_token"] = round(joules, 3)
        fields["energy_usd_per_1m_output_tokens"] = round(joules * 1e6 / 3.6e6 * kwh_usd, 4)
    return fields


def measure(model, qpc_path, case, args, tokenizer):
    devices = case["devices"]
    batch_size = case.get("batch_size", 1)
    prompts = [build_prompt(tokenizer, case["input_len"])] * batch_size

    def generate(generation_len):
        return model.generate(tokenizer=tokenizer, prompts=prompts, device_ids=devices, generation_len=generation_len)

    generate(min(case["generation_len"], 16))  # warm-up, loads the QPC; a short output keeps it cheap
    idle_w = read_board_power(devices)

    metrics = []
    with PowerSampler(devices) as sampler:
        for _ in range(args.repeats):
            metrics.append(generate(case["generation_len"]).perf_metrics)

    ttft = statistics.mean(float(m.prefill_time) for m in metrics)
    decode = statistics.mean(float(m.decode_perf) for m in metrics)
    total = statistics.mean(float(m.total_perf) for m in metrics)
    e2e = statistics.mean(float(m.total_time) for m in metrics)
    decode_w = sampler.peak()
    row = {
        **case_fields(case, args.model_name),
        "status": "ok",
        "repeats": args.repeats,
        "ttft_s": round(ttft, 3),
        # QEfficient reports TTFT per request for continuous batching, so this is one request's prefill rate
        "prefill_tok_s": round(case["input_len"] / ttft, 1),
        "decode_tok_s": round(decode, 2),
        "total_tok_s": round(total, 2),
        "e2e_s": round(e2e, 2),
        "decode_tok_s_per_stream": round(decode / batch_size, 2),
        "itl_ms": round(1000 * batch_size / decode, 2),
        "idle_board_w": idle_w,
        "decode_board_w": decode_w,
        "mean_board_w": sampler.mean(),
        "tok_s_per_w": round(decode / decode_w, 4) if decode_w else "",
        "tok_s_per_dollar": round(decode / args.card_price_usd, 4) if args.card_price_usd else "",
        "qpc": str(qpc_path),
    }
    row.update(cost_fields(row, args))
    return row


def run_case(case, args, tokenizer):
    model, qpc_path = load_and_compile(case, args)
    return measure(model, qpc_path, case, args, tokenizer)


def main():
    parser = argparse.ArgumentParser(description="K2 Horizon benchmark matrix on Cloud AI 100")
    parser.add_argument("--model-name", default="IFM/K2-Horizon-7B")
    parser.add_argument("--matrix", default=str(Path(__file__).parent / "matrices" / "benchmark_matrix.json"))
    parser.add_argument("--out", default="k2_horizon_benchmark.csv")
    parser.add_argument("--only", nargs="*", help="Run only these matrix ids")
    parser.add_argument("--repeats", type=int, default=3, help="Timed runs per case, after one warm-up")
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument("--card-price-usd", type=float, default=None, help="Card price for tokens/s per dollar")
    parser.add_argument("--card-hour-usd", type=float, default=DEFAULT_CARD_HOUR_USD, help="rental $ per card-hour")
    parser.add_argument("--electricity-usd-per-kwh", type=float, default=DEFAULT_ELECTRICITY_USD_PER_KWH)
    args = parser.parse_args()

    cases = json.loads(Path(args.matrix).read_text())
    if args.only:
        cases = [c for c in cases if c["id"] in args.only]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    out = Path(args.out)
    write_header = not out.exists()
    for case in cases:
        print(f"\n=== {case['id']} ===", flush=True)
        try:
            row = run_case(case, args, tokenizer)
        except Exception as exc:  # keep going, one failed case must not lose the others
            print(f"{case['id']} failed: {exc}", flush=True)
            row = failed_row(case, args.model_name, "failed", str(exc))
        with out.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()
                write_header = False
            writer.writerow(row)
        print(json.dumps(row, indent=1), flush=True)
    print(f"\nResults in {out}")


if __name__ == "__main__":
    main()
