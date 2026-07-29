# lpacr/analysis

Offline LP teşhisleri ve V5 kampanya raporu. GPU / eğitim / ağ yok; yalnız
kayıtlı atlas, validation bank ve TensorBoard event dosyaları.

## Tek komut — V5 raporu

```bash
.venv/bin/python -m lpacr.analysis.build_report
```

Çıktılar:

| dosya | içerik |
|---|---|
| `report/results.json` | Tüm sayılar (HTML buradan okur) |
| `report/v5_lp_analysis.html` | Self-contained Türkçe rapor |
| `V5_ANALIZ_BULGULARI.md` | HISTORY §12 adayı özet |

## Modüller

| dosya | ne yapar |
|---|---|
| `atlas.py` | multi-run registry, NaN-tolerant load, bootstrap index 0 kuralı |
| `validation_bank.py` | run #1 per-cell/replica bank + matched-design check |
| `scorecard.py` | A1/A3/A5/A4/A7 + C2 — her metrik null + CI |
| `noise_budget.py` | B1–B5 varyans bileşenleri / sınırlar |
| `interventions.py` | D1–D7 ölçülen kazanç veya «ölçülemedi» |
| `power_mde.py` | C4 run-to-run σ + MDE/güç tablosu |
| `tb_loader.py` | yerel TB skaler okuyucu |
| `lp_diagnostics.py` | §11.1–§11.6 tek-atlas CLI (koşu #4) |
| `sample_size_*.py` | §11.7 örnek büyüklüğü |
| `build_report.py` | hepsini birleştirir → results + HTML + MD |

## Eski CLI'lar (hâlâ çalışır)

```bash
.venv/bin/python -m lpacr.analysis.lp_diagnostics            # §11.1 – §11.6
.venv/bin/python lpacr/analysis/sample_size_power.py         # §11.7
```

## Tek kural

**Her teşhis kendi null'ı ile birlikte raporlanır.** Null'ı türetilmemiş bir sayı
kapı olarak kullanılamaz (§11.0).

## Bilinen sınır

- Run #1 kısmi şema: `performance_sem` yok → α_SEM yok (NaN doldurulur).
- Holdout / ued_validation yalnız run #1 yerelde; #2/#4 boş — uydurma yok.
- `atlas.lp_se()` gerçek LP gürültüsünü fazla tahmin edebilir (§11.7 anomaly).
- Tek seed: tüm LP−UNI farkları C4 gürültü tabanıyla okunmalı.
