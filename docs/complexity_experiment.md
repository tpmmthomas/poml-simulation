# Measuring the PoML complexity function

DeepProve's `bench.csv` reports `inference_time` and `prove_full` for each
sequence.  The latter includes witness generation and proving.  The online
work unit is therefore their sum; `context_generation_time` (PK/VK setup),
quantization, and `verify_full` are excluded.  Since DeepProve proves a single
causal pass without KV-cache decoding, its proof component is indexed by
`s=N+K` and may be stepwise because of padding and lookup domains.

Run a sweep (after the normal DeepProve build) with, for example:

```bash
cargo run --release --bin bench-llm -- --model gpt2 \
  --hf openai-community/gpt2 --sequence 16,32,64,128,256,512 \
  --load-params
python experiments/complexity_experiment.py .scratch/deep-prove/zkml/bench.csv
```

For a shared setup and independently chosen lengths, use the patched
benchmark's pair mode (for example, `--pairs 2:2,4:4,2:8,8:2`).  It creates
one variable-shape setup for the largest `N+K`, then executes every pair in
the same process.

For a larger 3-D campaign, run one command from the repository root:

```bash
python experiments/run_complexity_pairs.py --max-context 64 --pairs 100 --repeats 5
```

This generates 100 distinct pairs plus repeated boundary anchors (115 trials),
prints a conservative wall-time estimate, writes the raw CSV and fitted JSON below
`experiments/results/complexity/`, and saves a 3-D scatter/surface plot.  On
the measured 8-thread host this configuration is estimated at about 2.3 hours
(including setup), below a 4--5 hour budget.  Reduce `--pairs` to 80 if the
machine is slower; use `--seed` to reproduce the same design.

The current campaign command is:

```bash
python experiments/run_complexity_pairs.py --max-context 64 --pairs 100 --repeats 5
```

On a CUDA-capable host, use the GPU variant:

```bash
python experiments/run_complexity_pairs.py --cuda --max-context 64 --pairs 100 --repeats 5
```

This builds DeepProve with `--features cuda`; the output JSON records
`"cuda": true`.  GPU timings replace the CPU calibration but do not change
the architecture-level growth terms or the exclusion of setup and verifier
work.

The CUDA smoke timings suggest about 2.5 hours for the same 115-trial campaign
on this host.  Run the CPU and CUDA commands with the same `--seed` and
separate `--output-dir` values to compare their fits directly.

The runner automatically prepends `/usr/local/cuda/bin` to `PATH` and sets
`CUDA_HOME` when `--cuda` is selected, then passes `--fixed-length` and a
seeded `RNG_SEED` to the benchmark.

It runs 100 distinct pairs plus three measurements of five anchors, giving
115 trials.  Anchors are selected across the 7/8/9, 15/16/17, 31/32/33, and
63/64 boundaries.  `--fixed-length` disables EOS termination, and `RNG_SEED`
is set from `--seed` so prompts and quantisation samples are reproducible.
The CSV retains `inference_time`, `prove_claims`, `prove_commitment_opening`,
and `prove_full`; the JSON reports separate inference, proof, and combined
held-out fits.

Use fixed prompt lengths and fixed total sequence lengths so `K=s-N` is
known; repeat each cell after warm-up.  The fitting utility reports a least-
squares quadratic schedule in `s`, which is the identifiable variable for
this whole-sequence benchmark.  Reserve held-out `(N,K)` cells for validation.
If residuals cluster at length
boundaries, publish a deterministic length-bucket table instead of
extrapolating the polynomial.
