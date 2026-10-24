# qaic-llm-benchmark

LLM benchmarking on Qualcomm Cloud AI 100 Ultra through [QEfficient](https://github.com/quic/efficient-transformers). The first model set is the four dense K2 Horizon sizes (`IFM/K2-Horizon-0.9B`, `3.7B`, `7B`, `32B`).

This tool is kept out of the QEfficient repo on purpose. QEfficient prints `perf_metrics` for one run; everything here (case matrices, warm-up and repeats, board power, cost per token, CSVs, report) sits on top of it.

Benchmark a size only after it is onboarded and validated; the onboarding guide and its gates are in the [K2 Horizon onboarding README](https://github.com/Nouman945/efficient-transformers/tree/k2-horizon-onboarding-docs/examples/text_generation/k2_horizon).

## Setup

On the server, in the venv that has QEfficient installed (editable) from the fork branch with the K2 Horizon wrapper:

```csh
cd /local/mnt/workspace/$USER
git clone https://github.com/Nouman945/qaic-llm-benchmark.git
cd qaic-llm-benchmark
source <path to your QEfficient venv>/bin/activate.csh
python3 run_benchmark.py --dry-run
```

QEfficient comes from `Nouman945/efficient-transformers`, branch `k2-horizon-onboarding` (`pip install -e .` in that checkout). Server caches and csh setup are in [SERVER_SETUP.md](https://github.com/Nouman945/efficient-transformers/tree/k2-horizon-onboarding-docs/examples/text_generation/k2_horizon/SERVER_SETUP.md). `meta.json` in every results folder records the QEfficient commit and branch, this repo's commit and the SDK version.

## Results

One Ultra card, MXFP6 weights + MXINT8 KV cache (fp16 weights for the 0.9B), 128-token prompt, 256 generated tokens:

| Model | 1 device | 4 devices, single user | 4 devices, 16 users (aggregate / per user) | tok/s per W at 16 users |
|---|---|---|---|---|
| `IFM/K2-Horizon-32B` | does not fit | pending | pending | pending |
| `IFM/K2-Horizon-7B` | 14.1 tok/s | 48.6 tok/s, TTFT 0.06 s | 452 / 28 tok/s | 3.2 |
| `IFM/K2-Horizon-3.7B` | 23.4 tok/s | 65.9 tok/s, TTFT 0.04 s | 385 / 24 tok/s | 2.9 |
| `IFM/K2-Horizon-0.9B` | 45.8 tok/s | 134.9 tok/s, TTFT 0.03 s | 1070 / 67 tok/s | 9.1 |

The `client` workload (the customer's shapes) has not been run yet for any size; its rows go here when it has.

## Files

| File | Purpose |
|---|---|
| `run_benchmark.py` | the benchmark: runs every case for every model, writes CSVs and `REPORT.md` |
| `benchmark.py` | single-matrix engine used by `run_benchmark.py`; can be run on its own |
| `matrices/benchmark_matrix.json`, `matrices/benchmark_matrix_0_9b.json`, `matrices/benchmark_matrix_32b.json` | default cases, 128-token prompt (MXFP6 for 7B and 3.7B, fp16 for the 0.9B, 4-device cases only for the 32B) |
| `matrices/benchmark_matrix_client.json`, `matrices/benchmark_matrix_client_0_9b.json` | the cases to share with the customer: 1k/1k at 1, 16 and 32 users and a 10k-token prompt (0.9B: 1k/128 at 1 and 128 users, 1k/1k at 16) |
| `matrices/benchmark_matrix_serving.json`, `matrices/benchmark_matrix_serving_0_9b.json` | the full customer sheet: 1024-token prompts, long outputs, 1 to 32 users (128 for the 0.9B) |
| (no file) `--workload sweep` | the full grid for the shared report, built in `run_benchmark.py` (`sweep_cases`): context 2k to 32k, every prompt length that fits, 128 and 1024 output tokens, 1 to 32 users |

## Running

Same environment as the onboarding guide (`HF_HOME`, `QEFF_HOME`, `QEFF_LOG_PATH` on the local disk, venv active), from this repo's root, in tmux. Nothing else may run on the card during a benchmark.

```csh
git pull
python3 run_benchmark.py --dry-run                       # checks and plan, nothing compiled
python3 run_benchmark.py --workload client --repeats 1   # the cases to share with the customer
python3 run_benchmark.py                                 # 128-token prompts, 1 to 16 users
python3 run_benchmark.py --workload serving              # 1024-token prompts, long outputs, 1 to 32 users
python3 run_benchmark.py --workload client --models IFM/K2-Horizon-32B   # one size only
python3 run_benchmark.py --workload sweep --dry-run          # the sweep plan: cases, compiles, cases that do not fit
python3 run_benchmark.py --workload sweep                    # all four sizes, one CSV to share
python3 run_benchmark.py --workload sweep --resume           # after a stop: skips the cases already measured
```

### The sweep

One row per model x context x prompt x output x users, on the production precision (MXFP6 + MXINT8 KV, fp16 + MXINT8 KV for the 0.9B) and all 4 devices, prefill 1024:

| Context | Prompt lengths | Output lengths | Users |
|---|---|---|---|
| 2k | 1024 | 128, 1024 | 1, 4, 8, 16, 32 |
| 4k, 8k, 16k, 32k | 1024, half the context, context minus 1024 | 128, 1024 | 1, 4, 8, 16, 32 |

That is 130 cases per model but only 25 compiles: one QPC per (context, users), and every prompt and output length runs on it. The runner skips a case when the weights plus the KV cache of every user at full context are above 90% of the card's 128 GB (`MODEL_MEMORY`, `--no-fit-check` to try anyway); today that is the 32B at 32k with 32 users. Skipped and failed cases are still rows in the CSV (`status`, `error`), so the shared file shows what was not measured and why.

Expect days, not hours: the 32B compiles take 1 to 2 hours each. The sweep runs one warm-up (16 output tokens) and 1 timed run per case unless `--repeats` is given. Run one size at a time with `--models` if the card is needed in between, and `--resume` after any stop.

`k2_horizon_benchmark_sweep/k2_horizon_benchmark_all.csv` holds every model's rows and is rewritten after each model; `--report-only` rebuilds it and `REPORT.md` from the per-model CSVs.

Start with `client`: four cases per model on the production precision and all 4 devices, matching the customer's reference sheet (1024-in / 1024-out at 1, 16 and 32 users, plus a 10240-token prompt for TTFT; for the 0.9B 1024-in / 128-out at 1 and 128 users, plus 16 users at 1k/1k).

Why it is slow: every case is a separate compile (20 to 40 minutes for the 7B, 1 to 2 hours for the 32B), continuous-batching and single-stream cases need separate ONNX exports, and each case then runs 1 warm-up plus `--repeats` timed runs. `client` is about 2 hours per model (more for the 32B); the full `default` and `serving` sets take a day or more. Compiles are cached under `QEFF_HOME`, so a rerun with the same settings only regenerates.

Results: `/local/mnt/workspace/$USER/k2_horizon_benchmark[_client|_serving]/REPORT.md`, with the CSVs and `meta.json` (SDK version, QEfficient commit, settings) next to it. `--models`, `--only`, `--repeats`, `--card-price-usd` (fills tok/s per dollar) and `--report-only` narrow or rerun; `--help` lists them.

## The matrix files

Each case is one compile: `devices`, `mxfp6`, `mxint8_kv_cache`, `prefill_seq_len`, `ctx_len`, `input_len`, `generation_len`, `batch_size` (more than 1 = continuous batching). The `id` is the Case column of the report; read it as `<precision>_<devices>_ctx<context>[_in<prompt>][_cb<users>]`: `mx` is MXFP6 + MXINT8, `fp16` is fp16 weights, `4dev` is all four devices, `cb16` is 16 concurrent users.

The prompt is a prose text cut to exactly `input_len` tokens (`build_prompt` in `benchmark.py`), so TTFT is measured on real text, not padding.

## The report columns

`ttft_s` (time to first token), `decode_tok_s` (sum over all streams), `decode_tok_s_per_stream` (what one user sees), `itl_ms` (`1000 / per-stream tok/s`, a mean), `decode_board_w` (peak board power from `qaic-util -q`, 150 W TDP), `tok_s_per_w` (Performance per Watt), `tok_s_per_dollar` (needs `--card-price-usd`; agree the price with the customer).

Added for the shared report:

| Column | How it is computed |
|---|---|
| `status`, `error` | `ok`, `failed` (compile or run error) or `skipped` (estimated not to fit) |
| `prefill_tok_s` | `input_len / ttft_s`, one request's prompt rate |
| `mean_board_w` | mean board power over the timed runs (energy uses this, `decode_board_w` is the peak) |
| `joules_per_output_token` | `mean_board_w / total_tok_s` |
| `usd_per_1m_output_tokens` | `card_hour_usd / (total_tok_s * 3600) * 1e6`; `total_tok_s` is output tokens over the whole request, prefill included |
| `usd_per_1m_tokens` | same, counting prompt and output tokens |
| `energy_usd_per_1m_output_tokens` | `joules_per_output_token * 1e6 / 3.6e6 * electricity_usd_per_kwh`, card power only |
| `est_memory_gb` | weights + KV at full context for every user, the number the fit check uses |

Cost basis: Qualcomm does not publish a price per token; it quotes performance per watt and per TCO dollar, and rents the card through cloud partners. The default `--card-hour-usd 0.80` is Cirrascale's 8x AI 100 Ultra server at $4,699 a month (4699 / 8 cards / 730 hours); a case on fewer devices pays its share of the card. `--electricity-usd-per-kwh` defaults to 0.13, about the US commercial average. Both values are in every row and in `meta.json`. The cost columns are computed when a case is measured, so for another price recompute them in the spreadsheet from `total_tok_s`, `input_len`, `batch_size`, `e2e_s` and `mean_board_w` with the formulas above; `--report-only` does not change them.

Power by hand, in a second terminal during a decode run:

```csh
watch -n 1 /opt/qti-aic/tools/qaic-util -q
```

`Board power(Watts)` is about 48 W idle on this server and 120 to 130 W during a 4-device decode.
