# Topic 04 — Fixed validation bank + `best_spnte.pt` seçimi

**Bağımlılık:** Topic 01 (SPNTE metriği).
**Referans:** `../solun_plani.md` §10 (Değerlendirme, Fixed validation bank,
Checkpoint seçimi), §12 (kontrat testleri).

## Amaç

Ortak, dondurulmuş `84 × 48` validation bank'ı kur; her `200` PPO update'te
üretilen `model_200.pt … model_3000.pt` checkpoint'lerini bu bank üzerinde
offline değerlendirip en düşük macro-mean SPNTE'yi `best_spnte.pt` olarak seç.

## Neden offline

`solun_plani.md §10`, checkpoint'lerin **canlı training dağılımında değil** ortak
fixed bank'ta seçilmesini şart koşar. `save_interval=200` zaten mevcut
(`go2_v3_mlp_config.py:17`), yani periyodik checkpoint'ler var. Bu topic in-runner
seçim yazmaz; post-hoc bir seçim aracı yazar (runner'lara dokunmadan).

## Dokunulacak / yeni dosyalar

- `legged_gym/scripts/eval/ued_validation.py` (yeni) — 84×48 bank runner
  (mevcut terrain-eval altyapısını yeniden kullanır: controlled terrain-grid,
  fixed level, geometry-hash).
- `legged_gym/scripts/eval/select_checkpoint.py` (yeni) — macro-mean SPNTE +
  tie-break → `best_spnte.pt`.
- `configs/eval/v5_ued.yaml` (yeni) — validation bank spec, seed'ler, success
  eşiği, geometry hash pinleri.

## Fixed validation bank (§10)

- 84 hücre = `(5 tip × 4 seviye + 1 düz) × 4 v_x`.
- Hücre başına `48` deterministic replica → `84 × 48 = 4032` env.
- Replica başı bin-içi command draw önceden üretilir, tüm checkpoint/yöntemlerde
  aynen tekrar kullanılır.
- `validation_seed=31001` (seçim), `eval_seed=41001` (held-out final). Seed'ler ve
  geometry hash'leri headline'dan önce config'te dondurulur.
- Agregasyon: önce replica SPNTE → hücre ortalaması → 84 hücrenin **eşit ağırlıklı**
  macro-mean (kolay örnekler zor hücreleri ezmez).

## Checkpoint seçimi (§10)

- Ana kural: en düşük 84-task macro-mean SPNTE → `best_spnte.pt`.
- Tie-break (yalnız `1e-6` mutlak tolerans içinde eşitlikte):
  1. daha düşük worst-%10 task SPNTE (CVaR),
  2. daha düşük fall rate,
  3. daha yüksek success rate,
  4. daha erken iteration.
- Checkpoint'e yaz: `selection_metric=spnte_v1`, skor bileşenleri,
  validation-bank fingerprint, geometry hash'leri, `spnte_v_scale`, selected
  iteration. Resume sonrası `best_spnte` anahtarı/artifact'i korunur.
- `best_tracking.pt` ana seçim **değildir**; `model_3000.pt` yalnız provenance.

## Success kontratı (§10)

- Başlangıç: `≥ 900/1000` step survival **ve** `SPNTE_lin < 0.30`.
- Hücre başarısı tek rollout değil, önceden belirlenmiş replica-success oranı.
- `SPNTE_yaw < 0.30` yalnız Faz C'de yaw ekseni açılınca eklenir.
- Eşik sonuç görüldükten sonra değiştirilmez.

## Primary + secondary metrikler (§10)

- Primary: mean SPNTE, final success rate, sample-efficiency AUC, worst-%10 CVaR.
- Secondary: return/fall/survival, terrain-command slice heatmap, LP/probability
  zaman serisi, entropy/ESS/coverage, assignment-prob vs PPO transition occupancy
  (ayrı raporlanır — §10).

## Kabul testleri (§12)

- Her checkpoint tam `84 × 48` bank üzerinde ölçülmeden seçim yapılamaz.
- En düşük macro-mean SPNTE → `best_spnte.pt`; tie-break yalnız `1e-6` içinde.
- Resume `best_spnte` seçim anahtarını/artifact'ini kaybetmez.
- Geometry hash, command bank ve eval seed bütün yöntemlerde aynı.
- Assignment distribution ve PPO transition occupancy ayrı kaydedilir.

## Done tanımı

- `ued_validation.py` + `select_checkpoint.py` + `v5_ued.yaml` çalışır;
  `tests/test_ued_checkpoint_selection.py` yeşil.
- Sentetik checkpoint setinde doğru `best_spnte.pt` seçimi + tie-break doğrulandı.
- V4_terrain'e de aynı bank/seçim uygulanabilir (SPNTE ek sütun; §14.2 ile uyumlu).
