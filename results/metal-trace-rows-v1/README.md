# Historical Instruments interval rows

These CSVs are a new public projection of existing XML captures, not a new capture, benchmark or regenerated experiment. The unchanged [diagnostic receipt](../metal-residual-rmsnorm-v1/trace-diagnostic.json) identifies all original XML files by SHA256. Each input hash was checked before extraction.

- `native-application.csv` / `metal-application.csv`: 10,015 process-attributed application intervals each.
- `native-gpu.csv` / `metal-gpu.csv`: 5,007 process-attributed Compute intervals each.
- `manifest.json`: source XML hashes, CSV hashes and exact selected processes.
- CSV fields: zero-based source XML row index, process, category, start and duration in nanoseconds. Row ordering and integer times are retained; unrelated processes, thread details and unused labels are omitted.

The extraction uses `lab.metal_trace_summary.read_export` to resolve XML references, filters process names starting with `python (`, and for GPU rows requires channel `Compute`. There was exactly one matching PID in each arm, shared between its application and GPU records. Application category uses the existing parser's label prefix before `  (`. Original XML and full trace bundles remain local; CSV verification checks published rows and historical summary agreement, not independent acquisition authenticity or completeness of the unpublished XML.

```sh
python -m experiments.verify_metal_trace_rows
```

This read-only verifier checks hashes, exact process/category scope, and independently recomputes counts, medians, nearest-rank p95, GPU duration sums and observed spans against the unchanged receipt. CI runs it without GPU access. It does not measure Metal performance. The historical chain=50 benchmark remains authoritative; trace chain=1 and instrumentation perturb execution. `accepted=false`, default-disabled mode, missing bandwidth/occupancy counters and all failed gates remain unchanged. Neither interval counts nor the 16.67% logical-byte model establish actual kernel-dispatch reduction or a measured Roofline.
