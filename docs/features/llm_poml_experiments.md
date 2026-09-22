# LLM PoML experiments

The current design and runnable command are in
[Benchmark-backed experiments with measured length profiles](llm_benchmark_profiling.md).

The suite implements:

1. Honest liveness: 50 adopted blocks under sufficient demand.
2. Query selection: independently profiled short/long expected outputs, using
   the existing theoretical complexity C and held-out benchmark traces.
3. Collision-only completed complexity: an accelerated M-by-Q simulation
   crediting eligible response transactions.

All three now require a real benchmark bank. The earlier repeated-base-text
bank, artificial expected-length labels, and shared per-shape proof timings
are superseded. Their numerical results cannot support the revised design.
Experiment 3 still resamples measured runtimes; it does not run a new proof
for each simulated attempt. See the guide for the precise proof limitation,
metrics, outputs, commands, and migration details.
