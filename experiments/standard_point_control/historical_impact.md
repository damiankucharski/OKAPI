# Impact on earlier results from OKAPI/experiments

The scoped-postprocessing bug reproduced when an already trained binary adapter was asked for predictions after a multiclass fit in the same process. We inspected all 34 Git-tracked Python files under experiments at OKAPI revision db7b2b1, including the integration checks and the intentionally failing cross-task example. **No previously published performance measurement in this folder requires correction for this defect.** The inspection follows the archived programs and their documented command-line entry points. Unrecorded notebook sessions, downstream applications and private runs cannot be audited using these artifacts.

| Execution pattern | Paths inspected | Consequence for the archived study |
|---|---|---|
| Native GP with explicit binary postprocessing | fit_gap, guided, guided_equivalence, guided_stress, parity_operator, structural_key, ambiguous_groups, affine_heads, affine_suppressor, compositional_control | Binary fit/evaluation throughout; the experiment binds identity directly or through its native GP helper. |
| FusionBench OKAPIMethod used for tensor preparation or seeding | provider_refinement, survivor_reuse, survivor_workload, islands | The adapter seeds or serializes inputs for native GP; the engines use binary identity, including when comparing two optimizers. The paths do not retain a public adapter across different tasks. |
| Reuse scores, check algebra, or operate on frozen binary programs | guided_topn, guided_transfer, fit_gap/top_n_replay, islands/replay_baselines, matching_oracle and organic_fates | Either the procedure uses binary predictions throughout, or structural reconstruction is only for hashing. Other predictors here are arithmetic, Top-N, logistic or an explicitly constructed native tree. |
| Research notes for independent FusionBench panels | threshold_refinement, threshold_graft, gate_lineage | Measurement drivers and their trajectory/prediction replays live in FusionBench. Each published linked experiment uses binary probability channels. The FusionBench repository has its own impact audit. |
| Task-spanning fixtures and the defect witness | standard_point_control | These checks deliberately exercise both dimensions. The only recorded corrupt predictions are the labeled defect demonstration. |
| Standalone commands starting with process-default identity | operator_equivalence, raw_circuit, trimmed_pool, rng_audit | The first three measure binary cases only. The RNG test alternates two engines on the same binary inputs and default. A fresh-process replay verified these result files. |

## Direct replay

Each experiment was executed by its documented CLI in a fresh interpreter with one BLAS/OpenMP thread. The first three used the current OKAPI lab code and FusionBench d00e1a8. Every non-timing output field agreed **exactly** with its committed CSV string.

| Archive | Rows | Non-timing fields per row | Different fields |
|---|---:|---:|---:|
| operator_equivalence.csv | 480 | 14 | 0 |
| raw_circuit.csv | 192 | 13 | 0 |
| trimmed_pool_measurements.csv | 320 | 8 | 0 |
| trimmed_pool_stacker.csv | 96 | 8 | 0 |
| trimmed_pool_summary.csv | 20 | 9 | 0 |
| **Total** | **1,108** | **12,724 fields across all rows** | **0** |

For the RNG control, the current core still gives identical isolated and interleaved trajectories but has since changed its search implementation. Running its original audit script against revision 86e84ec with PYTHONHASHSEED 37 and 97 separately reproduced both recorded post-change files, with 16 hashes each. Running the same script with the earlier core 9871dac and hash seed 37 reproduced its 16 original pre-change hashes. Comparisons intentionally excluded elapsed times. The archives store scores and diagnostics for the replayed binary experiments; a historical process's global setting itself was never logged.

[Original script and result directory](../results) · [Standalone circuit and operator commands](../README.md) · [Trimmed-pool reproduction instructions](../trimmed_pool.md) · [Current defect witness and scoped fix](README.md)
