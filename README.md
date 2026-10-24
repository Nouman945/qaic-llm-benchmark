# qaic-llm-benchmark

LLM benchmarking on Qualcomm Cloud AI 100 Ultra through [QEfficient](https://github.com/quic/efficient-transformers): latency, throughput, board power and cost per token, for any text LLM QEfficient supports, written to one CSV you can share.

This tool is kept out of the QEfficient repo on purpose. QEfficient prints `perf_metrics` for one run; everything here (case grids, warm-up and repeats, board power, cost, memory-fit check, CSVs, report) sits on top of it.

Benchmark a model only after it is onboarded and its card output is validated. For the K2 Horizon sizes that is the [K2 Horizon onboarding README](https://github.com/Nouman945/efficient-transformers/tree/k2-horizon-onboarding-docs/examples/text_generation/k2_horizon).

## Setup

On the server, in the venv that has QEfficient installed (editable) with the model's wrapper:

```csh
cd /local/mnt/workspace/$USER
git clone https://github.com/Nouman945/qaic-llm-benchmark.git
cd qaic-llm-benchmark
source <path to your QEfficient venv>/bin/activate.csh
python3 run_benchmark.py --models IFM/K2-Horizon-7B --workload sweep --dry-run
```

For the K2 Horizon sizes, QEfficient comes from `Nouman945/efficient-transformers`, branch `k2-horizon-onboarding` (`pip install -e .` in that checkout). Server caches and csh setup are in [SERVER_SETUP.md](https://github.com/Nouman945/efficient-transformers/tree/k2-horizon-onboarding-docs/examples/text_generation/k2_horizon/SERVER_SETUP.md). `meta.json` in every results folder records the QEfficient commit and branch, this repo's commit, the SDK version and the model facts.

## Running

Nothing else may run on the cards during a benchmark. Run it in tmux.

```csh
# the dry run reads each model's config, prints its facts and the plan, and compiles nothing
python3 run_benchmark.py --models IFM/K2-Horizon-32B IFM/K2-Horizon-7B IFM/K2-Horizon-3.7B IFM/K2-Horizon-0.9B --workload sweep --dry-run

# the four K2 Horizon sizes, one CSV to share
python3 run_benchmark.py --models IFM/K2-Horizon-32B IFM/K2-Horizon-7B IFM/K2-Horizon-3.7B IFM/K2-Horizon-0.9B --workload sweep

# after any stop: skips the cases already measured
python3 run_benchmark.py --models IFM/K2-Horizon-32B IFM/K2-Horizon-7B IFM/K2-Horizon-3.7B IFM/K2-Horizon-0.9B --workload sweep --resume

# any other model; two cards; a smaller grid
python3 run_benchmark.py --models openai/gpt-oss-120b --workload sweep --devices 0-7 --ctx-lens 8192 32768 131072 --users 1 8 32

# hand-written cases instead of the grid
python3 run_benchmark.py --models IFM/K2-Horizon-7B --workload client --repeats 1
python3 run_benchmark.py --models Qwen/Qwen2.5-7B-Instruct --matrix my_cases.json
```

Results go to `<workspace>/qaic_benchmark[_<workload>]/`: `qaic_benchmark_all.csv` (every model, the file to share, rewritten after each model), one CSV per model, `REPORT.md` and `meta.json`. `--report-only` rebuilds the combined CSV and the report from the per-model CSVs. `--help` lists every flag.

## The sweep

One row per model x context x prompt x output x users. Defaults:

| Context | Prompt lengths | Output lengths | Users |
|---|---|---|---|
| 2k | 1024 | 128, 1024 | 1, 4, 8, 16, 32 |
| 4k, 8k, 16k, 32k | 1024, half the context, context minus 1024 | 128, 1024 | 1, 4, 8, 16, 32 |

Change the grid with `--ctx-lens`, `--users`, `--output-lens` and `--prefill-seq-len` (default 1024). Weights are MXFP6 and the KV cache MXINT8 unless `--precision fp16`, `--kv-cache fp16` or `models.json` says otherwise; devices are 0-3 (one card) unless `--devices` or `models.json` says otherwise.

The default grid is 130 cases per model but only 25 compiles: one QPC per (context, users), and every prompt and output length runs on it. The sweep runs one warm-up (16 output tokens) and 1 timed run per case unless `--repeats` is given. Expect days for a big model (the K2 32B compiles take 1 to 2 hours each); run one model at a time with `--models` if the card is needed in between, and `--resume` after any stop.

Cases that are not run still get a row (`status` `skipped`, with the reason in `error`), so the shared file shows what was not measured and why:

- the context is above the model's limit (`max_ctx_len` in `models.json`, else `max_position_embeddings` from the config);
- the weights plus the KV cache of every user at full context are above 90% of the devices' memory (32 GB per device). `--no-fit-check` tries them anyway.

A case that fails to compile or run gets `status` `failed` and the error.

## Model facts and models.json

The runner reads, for each model, the layers, KV heads, head dim and `max_position_embeddings` from the Hub config, and the parameter count from the safetensors metadata (no weights are downloaded). From those it estimates weights (2 bytes per parameter for fp16, 6.25 bits for MXFP6) and the KV cache (2 x layers x KV heads x head dim per token, 1 byte for MXINT8, 2 for fp16). For the four K2 Horizon sizes this gives exactly the size table of the onboarding README.

`models.json` holds the settings that differ from the flags, per model id. Every key is optional; a flag given on the command line wins.

| Key | Meaning |
|---|---|
| `precision` | `mxfp6` or `fp16` weights for the sweep (the K2 0.9B degrades with MXFP6) |
| `devices` | device ids for the sweep and for cases without `devices` |
| `max_ctx_len` | the longest context the QEfficient wrapper supports, when it is below the config (K2 Horizon: 65536) |
| `params_b` | parameter count in billions, when the safetensors count is wrong (gpt-oss: MXFP4 packs two weights per byte, so it reads 63B, not 117B) |
| `matrices` | case file per workload, when the model needs its own (`{"default": "benchmark_matrix_32b.json"}`) |
| `note` | free text, ignored |

A model that is not in `models.json` runs with the flags and defaults. If its config cannot be read (gated repo without a token, no network) the context cap and fit check are skipped with a warning.

## Case files

`matrices/` holds the hand-written workloads; `--workload` picks one (`default`, `serving`, `client`) and `--matrix` takes any file.

| File | Purpose |
|---|---|
| `benchmark_matrix.json` | 4k and 8k context, 128 to 4096-token prompts: fp16 and MXFP6, 1 and 4 devices, 1 to 16 users |
| `benchmark_matrix_client.json` | the cases to share with a customer: 1k/1k at 1, 16 and 32 users and a 10k-token prompt |
| `benchmark_matrix_serving.json` | 1024-token prompts, long outputs, 1 to 32 users |
| `benchmark_matrix_0_9b.json`, `benchmark_matrix_client_0_9b.json`, `benchmark_matrix_serving_0_9b.json` | the same for the K2 0.9B in fp16 (up to 128 users) |
| `benchmark_matrix_32b.json` | the default cases for the K2 32B, 4-device cases only |

Each case is one compile: `devices` (optional, else the model's devices), `mxfp6`, `mxint8_kv_cache`, `prefill_seq_len`, `ctx_len`, `input_len`, `generation_len`, `batch_size` (more than 1 = continuous batching). The `id` is the Case column of the report: `mx` is MXFP6 + MXINT8, `fp16` is fp16 weights, `4dev` is four devices, `c16` / `cb16` is 16 concurrent users.

The prompt is a prose text cut to exactly `input_len` tokens (`build_prompt` in `benchmark.py`), so TTFT is measured on real text, not padding.

## The columns

| Column | How it is computed |
|---|---|
| `status`, `error` | `ok`, `failed` (compile or run error) or `skipped` (context or memory) |
| `ttft_s` | time to first token; QEfficient reports it per request under continuous batching |
| `prefill_tok_s` | `input_len / ttft_s`, one request's prompt rate |
| `decode_tok_s` | decode tokens per second, summed over all users |
| `decode_tok_s_per_stream` | what one user sees |
| `itl_ms` | `1000 / per-stream tok/s`, the mean time between two tokens of one user |
| `total_tok_s`, `e2e_s` | output tokens over the whole request (prefill included) and its wall time |
| `idle_board_w`, `decode_board_w`, `mean_board_w` | board power from `qaic-util -q`: before, peak during and mean during the timed runs (150 W TDP per card) |
| `tok_s_per_w` | `decode_tok_s / decode_board_w` |
| `joules_per_output_token` | `mean_board_w / total_tok_s` |
| `usd_per_1m_output_tokens` | `card_hour_usd x num_devices / 4 / (total_tok_s x 3600) x 1e6`; a case pays for the share of the card it uses |
| `usd_per_1m_tokens` | the same, counting prompt and output tokens |
| `energy_usd_per_1m_output_tokens` | `joules_per_output_token x 1e6 / 3.6e6 x electricity_usd_per_kwh`, card power only |
| `tok_s_per_dollar` | `decode_tok_s / card_price_usd`, only with `--card-price-usd` |
| `est_memory_gb` | weights + KV at full context for every user, the number the fit check uses |

Cost basis: Qualcomm does not publish a price per token; it quotes performance per watt and per TCO dollar, and rents the card through cloud partners. The default `--card-hour-usd 0.80` is Cirrascale's 8x AI 100 Ultra server at $4,699 a month (4699 / 8 cards / 730 hours). `--electricity-usd-per-kwh` defaults to 0.13, about the US commercial average. Both values are in every row and in `meta.json`. The cost columns are computed when a case is measured, so for another price recompute them in the spreadsheet from `total_tok_s`, `input_len`, `batch_size`, `e2e_s` and `mean_board_w` with the formulas above; `--report-only` does not change them.

Power by hand, in a second terminal during a decode run:

```csh
watch -n 1 /opt/qti-aic/tools/qaic-util -q
```

`Board power(Watts)` is about 48 W idle on our server and 120 to 130 W during a 4-device decode.

## What it does not cover

- Text causal LMs only (`QEFFAutoModelForCausalLM`): no vision-language, embedding or speech models.
- One QPC per case: no disaggregated prefill/decode serving (the way QEfficient's gpt-oss examples run), no speculative decoding.
- The KV estimate counts full attention in every layer, so sliding-window models (gpt-oss) are over-estimated and may be skipped when they would fit; `--no-fit-check` tries them.

## Results so far

K2 Horizon, one Ultra card, MXFP6 weights + MXINT8 KV cache (fp16 weights for the 0.9B), 128-token prompt, 256 generated tokens:

| Model | 1 device | 4 devices, single user | 4 devices, 16 users (aggregate / per user) | tok/s per W at 16 users |
|---|---|---|---|---|
| `IFM/K2-Horizon-32B` | does not fit | pending | pending | pending |
| `IFM/K2-Horizon-7B` | 14.1 tok/s | 48.6 tok/s, TTFT 0.06 s | 452 / 28 tok/s | 3.2 |
| `IFM/K2-Horizon-3.7B` | 23.4 tok/s | 65.9 tok/s, TTFT 0.04 s | 385 / 24 tok/s | 2.9 |
| `IFM/K2-Horizon-0.9B` | 45.8 tok/s | 134.9 tok/s, TTFT 0.03 s | 1070 / 67 tok/s | 9.1 |

The sweep has not been run yet for any model.
