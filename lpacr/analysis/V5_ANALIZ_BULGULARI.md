# V5 LP-ACRL analiz bulguları (offline)

_Üretim: 2026-07-29T07:07:05.948767+00:00_

Bu not `HISTORY.md` §12 adayıdır. Eğitim yok; yalnız kayıtlı atlas +
validation bank + TB. Kural: **null'ı türetilmemiş sayı kapı olamaz.**

## Yönetici özeti

1. **LP güvenilirliği geç rejimde gürültü tabanında** — `run4 late α_min=0.0, α_SEM=0.0, null=0` → LP kapısını α=min(α_SEM,α_temporal) ile kapat; α≈0 ise uniform örnekle
2. **Uniform kolda da α_temporal düşük (temiz null) — sorun sampler rejiminden bağımsız** — `run1_uni late α_temporal=0.0326944948680844 (null 0)` → Ölçülemezlik görevin doğası / metrik doygunluğu; önce A7 doygunluk
3. **Eğitim-zamanı LP, held-out spnte iyileşmesini öngörmüyor (birincil test)** — `forward Spearman=0.08285916776349093 null=0 CI=[-0.2911410347271439, 0.24823326921129896] pass=False` → kriter geçerliliği yok → LP kovalamayı durdur veya tahminciyi değiştir
4. **vy/yaw sabitleme gürültüyü çözmez** — `within-cell command R²≈0.03608666493568099 (null 0)` → D3 düşük öncelik; kovaryat tavanı da düşük (D4)
5. **Daha çok episode geç rejimde bağlayıcı kısıt değil** — `σ²_signal late=-0.008468484022214118, 8× gain cells≈0.0` → episode bütçesini 8× şişirme; ufuk/metrik tarafına bak (D2/D6/D7)
6. **Stage birleştirme (D2) ölçülen α×excess eğrisi** — `best_k=4 value=0.21348631186961828` → optimum iç noktada olabilir; tepki gecikmesiyle trade-off
7. **Tek seed kampanya MDE'nin altında** — `With σ≈0.0332, detecting Δspnte=0.01 at 80% power needs ~173 seeds/arm; Δ=0.02 needs ~44/arm. Single-seed campaign advantages at 0.001–0.04 are mostly unresolvable.` → sonraki kampanya: σ≈0.03315597906475887; tabloya göre seed sayısı
8. **Kollar-arası LP korelasyonu (split-half alt sınır)** — `median corr=-0.009649245375797982 null=0.3467199075579466 CI=[-0.1346672051833419, 0.8956631517934366]` → düşükse LP ölçümü koşudan koşuya taşınmıyor
9. **Hücreler arası learnability dispersiyonu (A7)** — `final_cell_std=0.14948380351256907 between_cell_α=0.9959841250225502 near_noise=False` → cells still differ at final checkpoint beyond replica noise
10. **§11 bulguları multi-run'a taşındı** — `24 transfer kaydı` → yanlışlanan iddia yoksa kapı kuralı kalsın; UNI null ile genelle

## Skor kartı (özet)

| run | α_SEM late | α_temp late | lag1 excess | top10 |
|---|---|---|---|---|
| run1_lp | None | 0.05375512304744923 | 0.043064531915012416 | 0.0 |
| run1_uni | None | 0.0326944948680844 | 0.05220107621880482 | 0.0 |
| run2_lp | 0.005149167519105668 | 0.07281508917124913 | 0.12248088078561126 | 0.0 |
| run2_uni | 0.0464836156484038 | 0.057123485346864955 | 0.05327902449877586 | 0.0 |
| run3_crash | None | None | 0.5925988767360909 | 0.35 |
| run4_fixed | 0.0 | 0.09887042055645967 | 0.08266303347924603 | 0.0 |

## Gürültü bütçesi

σ_run (kampanya) ≈ **0.03315597906475887** (validation_bank_run1_late_macro_median_of=[late_sd=0.0042, |finalΔ|=0.0332, 5×paired_se=0.0356]).

With σ≈0.0332, detecting Δspnte=0.01 at 80% power needs ~173 seeds/arm; Δ=0.02 needs ~44/arm. Single-seed campaign advantages at 0.001–0.04 are mostly unresolvable.

## Müdahaleler (run4)

- `D2_longer_stage` rank=1: actionable_proxy_alpha_x_excess=0.21348631186961828 (null 0.0) — measure_on_curve_optimum_may_be_interior
- `D7_estimator_swap` rank=2: lag2_excess_over_null0=0.04856412147940405 (null 0.0) — switch_if_lag2_excess_improves
- `D5_pooling_factorization` rank=3: alpha_temporal_change_from_pooling=0.004362977732462414 (null 0.0) — do_not_pool_if_alpha_temporal_flat
- `D1_more_episodes` rank=4: delta_excess_cells_z_gt_2_per_stage_1x_to_8x=0.0 (null 0.0) — do_not_prioritize
- `D3_fix_vy_yaw`: ölçülemedi — validation bank only on run #1; not on disk for other runs
- `D4_covariate_residualization`: ölçülemedi — atlas has no per-episode commands; ceiling simulated on validation bank only (run #1)
- `D6_metric_swap`: ölçülemedi — per-cell episode length not in V5 atlas (reward-per-step blocked); validation bank only on run #1 for metric comparison

## §11 mutabakat

- §11.1 [run1_lp] doğrulandı: stage censoring ~46%, corr(late,perf)>0
- §11.1 [run1_uni] doğrulandı: stage censoring ~46%, corr(late,perf)>0
- §11.1 [run2_lp] doğrulandı: stage censoring ~46%, corr(late,perf)>0
- §11.1 [run2_uni] doğrulandı: stage censoring ~46%, corr(late,perf)>0
- §11.1 [run3_crash] genişledi: stage censoring ~46%, corr(late,perf)>0
- §11.1 [run4_fixed] doğrulandı: stage censoring ~46%, corr(late,perf)>0
- §11.2 [run1_lp] genişledi: #4: late α_SEM≈0, α_temporal≈0.1
- §11.2 [run1_uni] genişledi: #4: late α_SEM≈0, α_temporal≈0.1
- §11.2 [run2_lp] genişledi: #4: late α_SEM≈0, α_temporal≈0.1
- §11.2 [run2_uni] genişledi: #4: late α_SEM≈0, α_temporal≈0.1
- §11.2 [run3_crash] genişledi: #4: late α_SEM≈0, α_temporal≈0.1
- §11.2 [run4_fixed] genişledi: #4: late α_SEM≈0, α_temporal≈0.1
- §11.3 [run4_fixed] genişledi: pooling raises α_SEM not α_temporal
- §11.3 [run1_uni] genişledi: pooling raises α_SEM not α_temporal
- §11.3 [run2_uni] genişledi: pooling raises α_SEM not α_temporal
- §11.4 [] doğrulandı (teorik); run #4 atlas median was 0.414: lp_reliability pure-noise fixed point 0.444
- §11.6 [run1_lp] genişledi: signal horizon ~1 stage; lag1 excess ~0.08
- §11.6 [run1_uni] genişledi: signal horizon ~1 stage; lag1 excess ~0.08
- §11.6 [run2_lp] genişledi: signal horizon ~1 stage; lag1 excess ~0.08
- §11.6 [run2_uni] genişledi: signal horizon ~1 stage; lag1 excess ~0.08
- §11.6 [run3_crash] genişledi: signal horizon ~1 stage; lag1 excess ~0.08
- §11.6 [run4_fixed] genişledi: signal horizon ~1 stage; lag1 excess ~0.08
- §11.7 [run4_fixed] doğrulandı: sig2≈0 late; 8× budget buys few cells
- §11.7 [run2_lp] genişledi: sig2≈0 late; 8× budget buys few cells

## Yapılmadı

- **A6 full bandit/curriculum simulation** — partial: closed-form ceiling from α×horizon×cell-var only; full oracle bandit sim not run
- **E1 loop-gain full β surface** — partial: elasticity measured on #3/#4 where diagnostics exist; no offline β replay of crash
- **F1 diet-reweighted train reward** — yapılmadı — sebep: requires aligning TB iteration index with atlas stage diet; TB loaded for inspection but reweight not completed
- **F3 max-of-k checkpoint selection bias** — partial: see campaign.bank_noise macro curves; formal max-of-6 bias bound in findings
- **G1 V6 frontier full comparison** — yapılmadı — sebep: schema differs; only inventory entry present

## Yeniden üretme

```bash
.venv/bin/python -m lpacr.analysis.build_report
```

