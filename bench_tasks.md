
| Task | Eğitim dağılımı | Actor girdisi | Rol |
|---|---|---|---|
| `go2_bench_nodr` | DR kapalı | 45 proprioception | No-DR tabanı/floor |
| `go2_bench_mlp` | Dar DR | 45 proprioception | Ana memoryless DR baseline |
| `go2_bench_oracle_id` | Dar DR | 45 + gerçek 5D fizik `P` | `mlp` ile adil, dar-band oracle |
| `go2_bench_mlp_wide` | Geniş DR | 45 proprioception | Geniş oracle’ın adil kontrolü |
| `go2_bench_oracle` | Geniş DR | 45 + gerçek 5D fizik `P` | Geniş-band oracle |
| `go2_bench_mlp_rich` | Dar DR + PD gain + latency | 45 proprioception | Wave-2 rich-P kontrolü |
| `go2_bench_oracle_rich` | Aynı rich DR | 45 + gerçek 7D `P` | Wave-2 rich-P oracle |
