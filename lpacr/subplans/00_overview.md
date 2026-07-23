# V5 / UED — Subagent decomposition (overview)

Bu klasör, `../solun_plani.md`'yi bağımsız çalıştırılabilir iş kalemlerine böler.
Her dosya bir subagent tarafından yürütülecek şekilde self-contained yazıldı:
scope, dokunulacak dosyalar, diğer topic'lerle interface kontratı, dondurulmuş
kararlar, kabul testleri ve "done" tanımı.

Ana plan otoritedir; çelişkide `solun_plani.md §14` (Netleştirmeler) geçerlidir.

## Bağımlılık grafiği

```
01_spnte_metric ─────────────┐
                             ├──> 04_validation_and_checkpoint ──┐
02_ued_teacher_core ──┐      │                                   ├──> 05_arms_config_and_fazB
                      ├──> 03_genesis_integration ───────────────┘
                      │
   (01 ve 02 tamamen paralel; çekirdek env'e dokunmaz — önce bunlar)
```

| # | Konu | Bağımlılık | Risk | Çekirdek env'e dokunur? |
|---|---|---|---|---|
| 01 | SPNTE metrik + eval wiring (V4+V5) | — | düşük | hayır |
| 02 | Clean-room UED teacher | — | düşük | hayır |
| 03 | Genesis entegrasyonu + provenance | 02 | **yüksek** | evet (flag arkasında) |
| 04 | Validation bank + `best_spnte.pt` | 01 | orta | hayır |
| 05 | Kollar/config + kontrat + Faz B | 02,03,04 | orta | evet (config) |

## Ön koşul (her turdan önce, elle veya ilk subagent)

`genesis-wp/LeggedGym-Ex` bir git deposudur ve `lpacr/` artık içindedir. Turlardan
önce temiz baseline commit alınır:

```bash
cd /home/emin/code/online-estimation/genesis-wp/LeggedGym-Ex
git add -A && git commit -m "baseline: move lpacr into repo before v5/UED work"
```

Her subplan kendi feature branch'inde çalışır. 03 ve 05 çekirdek env/config'e
dokunduğu için değişiklikleri flag arkasına alır; flag kapalıyken v3/v4 birebir
korunmalı (`tests/test_v3_training_contract.py`, `tests/test_v4_training_contract.py`
yeşil kalmalı).

## Paylaşılan dondurulmuş kontratlar (bütün topic'ler uyar)

Bunlar `solun_plani.md §14`'ten türer; hiçbir subagent tek taraflı değiştirmez:

1. **Task space:** `(5 tip × 4 seviye + 1 düz) × 4 v_x = 84` moving task. Tip
   sırası taksonomi ile aynı (0 çıkan-merdiven … 4 rough, 5 düz). `TaskSpace`
   immutable, `fingerprint()` builder parametrelerinin tümünü kapsar.
2. **SPNTE default, eski metrikler crosscheck.** Payda `v_scale` dinamik
   (`max|lin_vel_x|`), artifact'e `spnte_v_scale` yazılır. First-fall
   accumulator, auto_reset açık (§14.3).
3. **Checkpoint schema** `solun_plani.md §9`'daki dict; `schema_version=1`.
4. **Standstill** per-env, `valid_for_curriculum=False`, LP/ALP'ye girmez.
5. **Command-curriculum + command_schedule** UED kollarında kapalı,
   `handcrafted_v4`'te açık (§14.5).

## Dispatch notu

01 ve 02'yi eşzamanlı iki subagent'a ver. 03, 02 bittiğinde başlar (interface'i
`02`'nin ürettiği `EpisodeCurriculum`/`TaskSpace`). 04, 01 bittiğinde. 05 en son,
diğer üçünü entegre eder ve Faz B eğitimini kurar. Her subagent kendi
`subplans/NN_*.md` dosyasını + `../solun_plani.md`'yi okur.
