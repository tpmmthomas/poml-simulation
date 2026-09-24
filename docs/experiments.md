# Running the Experiments

Run every command from the repository root. Use a new output directory for
each run. The commands implement the methods in the current paper; a reduced
run checks execution and semantics but does not reproduce a reported paper
number.

The experiments use three quantities throughout. An **inference–proof pair**
is one model inference together with its verified proof. `N` is the prompt
length, `K` is the generated length, and `C_{theta,Pi}(N,K)` is the public
complexity charged to the pair. A **target block time** is the calibration
target used to choose the lottery difficulty; it is not a guarantee that a
finite run will attain that mean.

## Setup

Install the base tools and experiment dependencies first:

```bash
pip install -e '.[dev,experiments]'
```

The fresh measurement and liveness experiments use GPT-2 and DeepProve:

```bash
pip install -e '.[deepprove,experiments]'
rustup toolchain install nightly-2026-01-27
python scripts/prepare_deepprove_protocol.py
```

The DDPM compatibility experiment uses Stable Diffusion:

```bash
pip install -e '.[diffusion,experiments]'
python scripts/download_sd_model.py
```

## Fresh inference–proof measurements

The liveness and wasted-work experiments begin with a bank of genuine GPT-2
inference–proof pairs. The default bank uses 32 WikiText-2 prompts, two fresh
replicates per prompt, and prompt lengths cycling through 8, 16, 24, and 32
tokens:

```bash
python experiments/measure_pairs.py --backend gpt2 --device cuda:0 \
  --queries 32 --replicates 2 --max-output 32 \
  --output experiments/results/measurements
```

Each replicate receives a new query binding, indexed inference VRF values, a
fresh model execution, and a fresh proof. The recorded duration includes online
inference and proving, but excludes setup and proof verification. The actual
generated length is recorded; it is not replaced by the requested output cap.
`completed.jsonl` is written after each pair and `measurements.json` is written
when the bank is complete. Synthetic smoke durations must not be called
measurements.

## Useful-work efficiency

`work_efficiency.py` measures the online cost of producing a fresh response and
computes the useful-work fraction proposed for the paper. It uses one
serialized context-64 GPT-2/DeepProve worker. The default paper campaign has
5 warm-up responses and 50 measured responses, with prompt lengths cycling
through 8, 16, 24, and 32 tokens. The command fixes the output cap at one token
to keep the full proof campaign practical:

```bash
python experiments/work_efficiency.py --backend gpt2 --device cuda:0 \
  --setup-directory experiments/results/llm_poml/protocol-context64-smoke/prover \
  --output experiments/results/work-efficiency \
  --queries 50 --warmup 5 --max-output 1 --bootstrap 10000 --seed 20260925
```

For each measured response, the script records the prover's inference,
proof-generation, and proof-verification times. These three components are
classified as useful work because the response and its correctness certificate
are the service delivered to the query author. It times the remaining online
work separately: inference and encryption VRFs, hashes, complexity evaluation,
encryption, signatures, serialization, artifact I/O, protocol validation, and
the lottery check. Model/setup initialization, query submission, network
communication, and duplicate-query races are outside this experiment.

The summary uses ratios of totals rather than averages of per-response ratios:

```text
alpha_cert = total_elapsed / total_useful
useful_fraction = total_useful / total_elapsed = 1 / alpha_cert
auxiliary_fraction = 1 - useful_fraction
```

The bootstrap resamples prompt clusters within each prompt-length stratum, so
repeated prompts in parallel shards are not treated as independent inputs.
`manifest.json` records hardware, source hashes, prover setup identity, and
the experimental substitutions: the model backend supplies a genuine model
proof, while the complete private PoML relation is a trusted-host receipt and
the VRF is the repository's sign-then-hash substitute. These substitutions are
part of the
reported implementation boundary and should not be read as a benchmark of a
production NIZK or formally unbiasable VRF.

To reanalyze a completed run without creating new proofs:

```bash
python experiments/work_efficiency.py --summarize \
  experiments/results/work-efficiency --bootstrap 10000 --seed 20260925
```

Independent GPU workers may be used when their device allocation is separated.
For a 50-response campaign, launch three workers with 17, 17, and 16 measured
responses (and two warm-ups each), then merge their completed output:

```bash
python experiments/work_efficiency.py --device cuda:0 --queries 17 --warmup 2 \
  --max-output 1 --seed 20260925 \
  --setup-directory experiments/results/llm_poml/protocol-context64-smoke/prover \
  --output experiments/results/work-efficiency-shard0
python experiments/work_efficiency.py --device cuda:1 --queries 17 --warmup 2 \
  --max-output 1 --seed 20260925 \
  --setup-directory experiments/results/llm_poml/protocol-context64-smoke/prover \
  --output experiments/results/work-efficiency-shard1
python experiments/work_efficiency.py --device cuda:2 --queries 16 --warmup 2 \
  --max-output 1 --seed 20260925 \
  --setup-directory experiments/results/llm_poml/protocol-context64-smoke/prover \
  --output experiments/results/work-efficiency-shard2
python experiments/merge_work_efficiency.py \
  experiments/results/work-efficiency-shard0 \
  experiments/results/work-efficiency-shard1 \
  experiments/results/work-efficiency-shard2 \
  --output experiments/results/work-efficiency
```

The merge recomputes the ratio of aggregate useful and elapsed time; it does
not average shard-level ratios. The measured campaign separates GPU devices;
its manifest records that the workers inherited the shared host CPU affinity.
For strict CPU isolation, wrap each worker with a disjoint `taskset -c` range
before launching it.

## PoML Liveness and Block Generation Stability

This experiment uses four miners, a pool of 256 queries, 50 blocks, and a
300-second target. The default mode replays measured `(T,C)` pairs, where `T` is
the recorded inference–proof time and `C` is its complexity. It samples the
pair jointly, so time and complexity from one measurement are never separated:

```bash
python experiments/liveness.py \
  --measurements experiments/results/measurements/measurements.json \
  --output experiments/results/liveness
```

The replay does not create new proofs. It draws an independent 256-bit lottery
value, applies the exact rounded complexity threshold, and lets each virtual
miner process a uniformly shuffled query pool. Query demand is replenished
between blocks.

The same command also produces a Bitcoin-style double-SHA-256 baseline. This
is a hash lottery, not a simulation of Bitcoin's transaction or network
protocol. To use the statistical control instead of actual hashing, add
`--pow-mode poisson`.

To execute fresh model proofs during the virtual race, use a matching GPT-2
measurement bank and setup:

```bash
python experiments/liveness.py --execution fresh --backend gpt2 --device cuda:0 \
  --measurements experiments/results/measurements/measurements.json \
  --setup-directory experiments/results/measurements/proofs/setup \
  --max-output 32 --output experiments/results/liveness-fresh
```

Fresh proving is serialized on the host. Virtual miners still advance by their
measured service durations, while physical work already performed by canceled
attempts is reported separately. Setup, verification, signatures, encryption,
and network delay are not part of the virtual service time.

For a quick event-loop check:

```bash
python experiments/liveness.py --smoke --blocks 3 --pow-mode poisson \
  --output experiments/results/liveness-smoke
```

## Wasted Work Analysis

This experiment asks how much completed inference–proof work is duplicated
before the first winning completion. Each miner samples a query permutation
without replacement from a fixed pool. Measured `(T,C)` pairs are pooled and
resampled jointly; no new model inference, proof, or ciphertext is created by
the replay.

The default grid is the one used in the paper:

| Quantity | Values |
| --- | --- |
| Miners `M` | 10, 20, 50, 100, 200, 500, 1,000, 2,000, 5,000, 10,000 |
| Query pool `Q` | 20, 50, 100, 200, 500, 1,000, 2,000, 5,000, 10,000, 20,000, 50,000, 100,000 |
| Target block time | 300, 600, 900 seconds |
| Repetitions | 100 per configuration |

Run it with:

```bash
python experiments/wasted_work.py \
  --measurements experiments/results/measurements/measurements.json \
  --output experiments/results/wasted-work
```

Let `A` be all completed pairs through the first winning completion, including
duplicates, and `F` be the first completion for each distinct query. The
reported wasted work is

```text
W = (sum(C over A) - sum(C over F)) / sum(C over A).
```

Unfinished attempts are excluded. Runs that exhaust the finite query pool
before a winner remain in `runs.csv`; they are not treated as successful blocks
and are excluded from adopted-race means. `cells.csv` and the heatmaps report
the conditional mean and sample standard deviation for adopted races.

```bash
python experiments/wasted_work.py --smoke --miners 2,4 --queries 4,20 \
  --targets 300 --repeats 3 --output experiments/results/waste-smoke
```

### Additional Wasted-Work Results

The three target times produce the paper's additional heatmaps at 300, 600,
and 900 seconds. They are outputs of **Wasted Work Analysis**, not a separate
kind of experiment.

## DDPM Compatibility: Formal Statements and Experiments

This experiment evaluates the computational-independence intuition for Stable
Diffusion v1-4. It records the pre-denoiser latent `x_t` at every reverse step;
it does not claim to prove a full Stable Diffusion model inside the PoML
protocol.

```bash
python experiments/diffusion_compatibility.py \
  --model models/stable-diffusion/stable-diffusion-v1-4-fp16 \
  --device cuda:0 --output experiments/results/diffusion-compatibility
```

The defaults are 1,000 base inputs, 50 **DDPM** steps, 512-equivalent latent
size, and perturbation scales `sigma = 0.001, 0.01, 0.1, 0.5, 1.0`. For each
perturbation, the base and perturbed trajectories share the reverse-step noise;
the independent baseline has independent initial and reverse-step noise. The
outputs report per-step cosine similarity and the minimum `L_infinity` distance
over all samples and steps.

A small executable check is:

```bash
python experiments/diffusion_compatibility.py --samples 1 --steps 4 \
  --size 256 --sigmas 0.001 --output experiments/results/diffusion-smoke
```

## LLM Compatibility: Formal Statements and Experiments

The LLM experiments use the pinned GPT-2 small checkpoint in float32 evaluation
mode. Only original prompt embeddings receive independent Gaussian noise. The
noise scale is `sigma_abs = alpha * s_E`, where `s_E` is the population standard
deviation of the frozen token-embedding table. Generated-token embeddings stay
clean.

### Preservation of benchmark utility

Run the utility experiment on WikiText-2, HellaSwag, PIQA, and ARC-Easy:

```bash
python experiments/gpt2_compatibility.py utility --device cuda:0 \
  --output experiments/results/gpt2-utility
```

The defaults use 100 examples per task, one clean baseline, three noisy
replicates, and `alpha = 0.05, 0.1, 0.2, 0.4`. WikiText-2 is scored by
teacher-forced perplexity. The other tasks use minimum summed continuation
negative log-likelihood. Accuracy is stored as a fraction in JSON.

### Computational independence of LLMs

Run the trace experiment as follows:

```bash
python experiments/gpt2_compatibility.py trace --device cuda:0 \
  --output experiments/results/gpt2-trace
```

It uses the same four tasks plus LAMBADA and Resisting Correction, 100 prompts
per task, four independent challenge pairs, and prefixes `0, 1, 4, 8, 16, 32`.
The two challenged runs decode independently at temperature 1, top-k 50, and
top-p 0.95, stopping when either run reaches EOS. The experiment compares
context matrices, attention and feed-forward outputs, final normalization, and
the last-position logit vector. It records the actual number of comparisons;
not every pair reaches every prefix.

Use `--manifest prompts.json` to reuse exactly tokenized examples across runs.
Small checks can use `--examples 1 --alphas 0.05 --replicates 1` for utility or
`--examples 1 --alphas 0.05 --pairs 1 --steps 0,1,4` for traces.

## A Reference Complexity Function for GPT-2 and DeepProve

The complexity function makes the cost of an inference–proof pair public. It
uses `S=N+K`, attention work `U`, cache work `R`, the GPT-2 reduction term
`D(S)`, and a padded-length proof vector `B_P`, where
`P = 2^ceil(log2(S))`. The default weights are uniform.

The offline calculator needs no model or GPU:

```bash
python experiments/complexity_counts.py --calculate 16:8
python experiments/complexity_counts.py --plan-only
```

The validation plan covers 40 distinct `(N,K)` pairs and 24 total lengths. To
rebuild the ledger from instrumented DeepProve executions:

```bash
python experiments/complexity_counts.py --prepare --cuda --device 0 \
  --output-dir experiments/results/complexity-counts
```

Inference counts are checked exactly. Proof-operation counts are compared with
the declared componentwise tolerance; this is an implementation check, not a
claim about the paper's maximum observed error. Setup and proof verification
are excluded, and the reference schedule describes the paper's autoregressive
graph rather than every extra noise or sampling operation in the modified
worker.

Fit nonnegative runtime weights from a completed genuine measurement bank:

```bash
python experiments/fit_complexity.py \
  experiments/results/measurements/measurements.json \
  --output experiments/results/complexity-fit
```

The fit uses prompt-grouped validation so duplicate prompts never cross folds.
It reports feature rank and held-out error against the uniformly weighted
baseline. Correlated operation weights are not individually identifiable; the
exported weights are a calibration choice, not a new ticket count.

## Reading results

Every run writes a manifest or `run.json` with its backend, model/setup digest,
seed, and metric scope. Keep reduced checks separate from paper-scale output.
The [verification record](verification.md) lists the checks already completed
and the campaigns that were not rerun because they require substantial model,
GPU, or CPU resources.
