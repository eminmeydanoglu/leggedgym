# Topic 01 — SPNTE metriği + eval wiring (V4 + V5)

**Bağımlılık:** yok. İlk başlayabilir, çekirdek env'e dokunmaz.
**Referans:** `../solun_plani.md` §10 (SPNTE) ve §14.1–14.3.

## Amaç

Stability-Penalized Normalized Tracking Error (SPNTE) metriğini eval harness'ına
ekle. Default raporlama metriği SPNTE olsun; mevcut metrikler crosscheck için
korunsun ve aynı koşumda birlikte hesaplansın. V4_terrain kampanyasında ek sütun
olarak, V5'te primary olarak çalışsın.

## Dokunulacak dosyalar

- `legged_gym/scripts/eval/metrics.py` — `MetricAccumulator`'a first-fall SPNTE
  state'i ekle.
- `legged_gym/scripts/eval/v3_eval.py` — `MetricAccumulator(...)` çağrısına
  `v_scale` geç (satır ~381); `compute()` çıktısındaki yeni SPNTE alanlarını
  payload'a yaz (~393). `v_scale`, o koşumun command bankasından
  (`bank["lin_vel_x"]`, ~327) `max(abs(lo), abs(hi))` ile türetilir.
- `configs/eval/v4_terrain.yaml` — dokunmadan çalışmalı (metrik ek sütun; ama
  raporlama/aggregation SPNTE'yi de bassın; report tarafına ekle).
- Test: `tests/test_spnte_metric.py` (yeni).

## Frozen kararlar

- `SPNTE_lin = ( Σ_{t<k_f} e_t + (K - k_f) ) / K`,
  `e_t = clip(|v_x_cmd - v_x_base| / v_scale, 0, 1)`.
- `k_f` = ilk `done & ~time_out` adımı (per-env). Düşme yoksa `k_f = K`.
- `v_scale = max(|lin_vel_x_min|, |lin_vel_x_max|)`, koşum config'inden dinamik;
  hardcode yok. Artifact'e `spnte_v_scale` yaz.
- **auto_reset AÇIK kalır.** SPNTE, ilk düşüşten sonraki adımları saymaz ve kalan
  horizon'u `1.0` ile doldurur; ama eski metrikler tüm horizon'u tüketmeye devam
  eder (ikisi aynı stream'den beslenir). Auto_reset'i KAPATMA — eski metriklerin
  eşzamanlı koşumunu bozar.
- `K` = harness'ın gerçekten koştuğu adım sayısı (`self._steps`), sabit 1000
  varsayma; horizon config'ten gelir.
- `SPNTE_yaw` de aynı mantıkla (`ang_err`, `v_scale_yaw = max|ang_vel_yaw|`)
  loglanır ama V5 Faz A/B checkpoint seçimine karışmaz (§10).

## MetricAccumulator değişiklik iskeleti

Constructor'a `v_scale: float` (ve opsiyonel `v_scale_yaw`) ekle. Yeni buffer'lar:

- `_first_fall_step` (long, init `-1`) — per-env ilk `fall` adımı.
- `_spnte_err_sum` (float) — `t < k_f` iken `e_t^{lin}` toplamı.
- (yaw için simetrik `_spnte_yaw_err_sum`.)

`update()` içinde, `fall` maskesi zaten hesaplanıyor. Her adımda:

- `still_first = _first_fall_step < 0` (henüz düşmemiş env'ler);
- `_spnte_err_sum += where(still_first, clip(lin_err / v_scale, 0, 1), 0)`;
- `_first_fall_step = where(fall & still_first, _steps, _first_fall_step)`.

`compute()` içinde:

- `k_f = where(_first_fall_step < 0, K, _first_fall_step)` (K = `self._steps`);
- `spnte_lin = (_spnte_err_sum + (K - k_f)) / K`.

Çıktı dict'ine ekle: `spnte_lin`, `spnte_yaw`, `first_fall_step`. Mevcut anahtarlar
(`tracking_lin_err`, `fall_rate`, ...) aynen kalır.

> Not: `lin_err` harness'a `|v_x_cmd - v_x_base|` olarak besleniyor (v3_eval'de
> `lin`). SPNTE yalnız `v_x` eksenini kullanır; harness lin_err'i xy-norm olarak
> veriyorsa, SPNTE için ayrı bir skaler v_x-error beslemek gerekir — bunu
> doğrula ve gerekiyorsa `update(..., lin_x_err=...)` ek argümanı ekle.

## Kabul testleri (`tests/test_spnte_metric.py`)

`../solun_plani.md §12`'den:

1. Düşme olmayan sabit-hatalı sentetik episode'da `spnte_lin ==` normalize tracking
   hatası (`e_t` sabit → ortalaması).
2. İlk düşüşten sonraki auto-reset adımları SPNTE'yi **iyileştirmez**: `k_f`
   sonrası düşük-hata besle, skor değişmesin (`(Σ_{t<k_f} + (K-k_f))/K`).
3. `v_x_cmd = 0` dahil tüm destekte SPNTE finite ve `[0,1]`.
4. `v_scale` dinamik: `[-1,1]` bankasında `v_scale=1.0`, `[0,2]`'de `2.0`;
   `spnte_v_scale` artifact alanı doğru.
5. Eski metrikler bozulmadı: aynı sentetik stream'de `tracking_lin_err`,
   `fall_rate` değerleri değişiklik öncekiyle aynı (regression guard).

## Done tanımı

- `test_spnte_metric.py` yeşil; mevcut eval testleri (`tests/test_eval_v2.py`,
  `tests/test_v3_eval.py`) yeşil.
- V4_terrain raporunda SPNTE yeni sütun olarak görünüyor, eski sayılar değişmemiş.
- `v_scale` hiçbir yerde hardcode değil; artifact'te `spnte_v_scale` var.
