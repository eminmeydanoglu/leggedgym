# v5 UED / LP-ACRL Deney Tarihçesi

**24 Temmuz 2026 (Cuma) – 27 Temmuz 2026 (Pazar)**
Go2 quadruped, Genesis + legged_gym + rsl_rl, PPO, 4096 env, UHeM Altay `a128` (hesap `kds3by`).

Bu belge dört ana koşuyu, aralarındaki analizleri, bulunan iki üretim bug'ını ve tüm sayısal sonuçları kayıt altına alır. Bütün sayılar kümedeki gerçek log/atlas dosyalarından okunmuştur; hiçbiri tahmin değildir.

---

## 0. Deney kurgusu

### 0.1 Görev uzayı

84 hücre = **4 vx_bin × 21 terrain_cell**, indeksleme:

```
task_id = vx_bin * 21 + terrain_cell
```

| eksen | değerler |
|---|---|
| `vx_bin` (4) | `0.2–0.5`, `0.5–1`, `1–1.5`, `1.5–2` m/s |
| `terrain_cell` (21) | 0–3 `stairs_up` L1–L4 · 4–7 `stairs_down` L1–L4 · 8–11 `slope_up` L1–L4 · 12–15 `slope_down` L1–L4 · 16–19 `rough` L1–L4 · 20 `flat` L1 |

Task-space fingerprint (tüm koşularda aynı): `dc10826138afc720cf828dd332f9415269cffd934a77054a464a6a0c451e2bce`

### 0.2 LP-ACRL formülasyonu (Li/Li/Hutter, arXiv 2601.17428)

```
Eq. 6:   LP_cj(ζ) = R_cj(ζ) − R_c(j−1)(ζ)
Eq. 7:   p(ζ) = softmax( LP(ζ) / β )        # her stage'de sıfırdan kurulur
```

Kritik nokta: **makale hiçbir koruma mekanizması kullanmıyor.** Çıplak softmax, her stage'de rebuild, olasılık stage'ler arasında taşınmıyor. **β makalede hiçbir yerde açıklanmamış** — bizim kalibre etmemiz gereken tek serbest parametre buydu, ve bu belgedeki hikâyenin tamamı o kalibrasyonun hikâyesi.

### 0.3 Ölçüm metrikleri

| metrik | anlamı |
|---|---|
| `macro_mean_spnte_lin` | ana metrik; **düşük = iyi** |
| `macro_success_rate` | replika bazında başarı |
| `cell_success_rate` | 84 hücrenin kaçı "başarılı" sayıldı |
| `macro_fall_rate` | düşme oranı |
| `worst_10pct_cvar_spnte_lin` | en kötü %10'un CVaR'ı (kuyruk riski) |

İki ayrı bank: **validation** (checkpoint seçimi için) ve **holdout** (final rapor). Bank fingerprint'leri tüm koşularda aynı.

### 0.4 Sampler sağlık teşhisleri (atlas)

| teşhis | tanım |
|---|---|
| `effective_sample_size` (ESS) | `1/Σpᵢ²` — 84 = tam uniform, 1 = tek hücre |
| `max_cell_probability` | en sıcak hücrenin olasılığı |
| `tv_distance_uniform` | `0.5·Σ|pᵢ − 1/N|` |
| `top10_overlap_prev` | ardışık stage'lerin ilk-10 sıcak hücre örtüşmesi — **kalıcı frontier var mı** |
| `lp_reliability_median` | `|LP| / lp_sem` medyanı — **ölçüm SNR'ı** |
| `eligible_fraction` | LP'sine güvenilen hücre oranı (episode kapısını geçenler) |
| `sampled_lp_mass` | güvenilen pozitif LP'nin ne kadarını hedefliyoruz (uniform referansı = 1/84 = 0.0119) |

---

## 1. Koşu #1 — 24 Temmuz Cuma, `fixed β = 5`

**Atlas:** `506791_20260724_135345` · **Model dizinleri:** `Jul24_13-55-22_v5_lp_acrl_genesis_seed1`, `Jul24_13-55-38_v5_uniform_genesis_seed1`

### Konfig

```python
V5_STAGE_LENGTH_CONTROL_STEPS = 2000
V5_BETA    = 5.0
V5_EPSILON = 0.0
# temperature_mode, max_cell_probability, min_stage_episodes_for_lp henüz yok
```

(Bu koşudan önce 24 Tem 10:34 ve 13:12–13:45 arasında seed 17/23/31/41/43 ile bir dizi başarısız başlatma denemesi var — `model_1.pt`, `model_5.pt`, `model_90.pt` gibi erken kesilmiş dizinler. Bunlar veri üretmedi.)

### Sampler davranışı (atlas, 36 frame, step 2000–72001)

| | LP-ACRL | uniform |
|---|---|---|
| ESS | 71.81 – 84.00 (medyan **82.27**) | 84.00 sabit |
| maxp | 0.01 – 0.03 (medyan 0.02) | 0.012 |

**LP kolu fiilen uniform örnekledi.** 84 hücrelik uzayda ESS 82 demek, "curriculum" diye bir şeyin olmadığı demek.

### Validation eğrisi

| iter | LP spnte | LP succ | LP cellsucc | | UNI spnte | UNI succ | UNI cellsucc |
|---|---|---|---|---|---|---|
| 1000 | 0.2219 | 0.783 | 0.607 | | 0.4686 | 0.411 | 0.262 |
| 1400 | 0.1790 | 0.820 | 0.667 | | 0.3060 | 0.692 | 0.500 |
| 1800 | 0.1519 | 0.854 | 0.702 | | 0.2003 | 0.821 | 0.702 |
| 2200 | 0.1493 | 0.866 | 0.702 | | 0.1866 | 0.836 | 0.702 |
| **2600** | **0.1265** | 0.902 | 0.798 | | **0.1680** | 0.854 | 0.714 |
| 3000 | 0.1397 | 0.886 | 0.786 | | 0.1729 | 0.838 | 0.667 |

### Sonuç — ve yanılgı

Görünürde **LP 0.1265 vs uniform 0.1680**, yani büyük bir LP avantajı. Bu, projenin en heyecan verici anıydı ve **yanlıştı.**

- Bu koşu için **holdout eval hiç yapılmadı** — yukarıdaki sayılar validation bank'ından, checkpoint seçim eğrisinden. `Jul24_13-55-*` dizinlerinde `heldout_final/` yok.
- Sampler ESS 82 ile uniform gibi örneklediğine göre, iki kol arasında **mekanizma farkı yoktu.** O halde 0.042'lik fark bir mekanizmadan gelemez.
- Cuma'nın **uniform kolu outlier'dı**: iter 1000'de spnte 0.4686 / success 0.411 ile başladı, iter 1000'den itibaren sürekli geride kaldı, atlas macro-perf 8–9 seviyesinde takıldı (diğer üç koşuda 10+).

Yani Cuma'daki "avantaj", tek-seed varyansıydı. Bu, sonraki round'da doğrudan test edildi.

---

## 2. Koşu #2 — 27 Temmuz sabah, `adaptive_ess`

**Atlas:** `507739_20260727_100500` (step 6000–52001) + `507758_20260727_110022` (step 54001–82001) — iki resume dikişi
**Model dizinleri:** `Jul27_11-01-35_v5_lp_acrl_genesis_seed1`, `Jul27_11-01-19_v5_uniform_genesis_seed1`

### Konfig

```python
V5_STAGE_LENGTH_CONTROL_STEPS = 2000
V5_BETA                  = 2.5
V5_EPSILON               = 0.03
V5_TEMPERATURE_MODE      = "adaptive_ess"
V5_BETA_MIN              = 0.75
V5_BETA_MAX              = 8.0
V5_BETA_EMA              = 0.8
V5_TARGET_ESS_RATIO_MIN  = 0.5
V5_MAX_CELL_PROBABILITY  = 0.08
V5_MIN_STAGE_EPISODES_FOR_LP = 16
V5_CONFIDENCE_SCALE      = 1.0
```

Fikir: β'yı sabitlemek yerine, ölçümün ne kadar güvenilir olduğuna göre hedef bir ESS belirle ve β'yı ona göre ayarla:

```
target_ess    = N − signal_quality · (N − 0.5·N)
signal_quality = median(reliability) × eligible_fraction
```

### Sampler davranışı

| | LP-ACRL | uniform |
|---|---|---|
| ESS | 57.06 – 81.59 (medyan ~73–75) | 84.00 sabit |
| maxp | 0.01 – 0.06 (medyan 0.03) | 0.012 |
| `effective_beta` | 0.92 – 3.47 (medyan ~1.2) | — |
| `target_ess` | 56.74 – 78.58 (medyan ~67) | — |
| `signal_quality` | 0.13 – 0.65 (medyan ~0.40) | — |
| `eligible_fraction` | 0.29 – 1.00 (medyan 0.99) | — |

**Yine fiilen uniform.** Sebep açık: `signal_quality` hiç 0.65'i geçmedi, dolayısıyla `target_ess` hep 65–79 bandında kaldı. Formül, ölçüm gürültülüyken uniform'a yaklaşmayı emrediyordu — ve ölçüm **her zaman** gürültülüydü.

### Validation eğrisi

| iter | LP spnte | | UNI spnte |
|---|---|---|---|
| 2400 | 0.1465 | | 0.1410 |
| 2800 | 0.1568 | | **0.1387** ← seçildi |
| 3200 | 0.1423 | | 0.1522 |
| **3500** | **0.1393** ← seçildi | | 0.1395 |

### Holdout final — asıl sonuç

| kol | seçilen iter | **spnte** | succ | cellsucc | fall | CVaR |
|---|---|---|---|---|---|---|
| LP-ACRL | 3500 | **0.1295** | 0.8889 | 0.7619 | 0.0804 | 0.4639 |
| uniform | 2800 | **0.1306** | 0.8919 | 0.7262 | 0.0823 | 0.4655 |

**Fark 0.0011. Berabere.**

Cuma'daki 0.042'lik "avantaj" böylece açıklandı: tek-seed arm-içi varyans, iddia edilen etkiyle aynı büyüklükte. Bu, projenin en önemli metodolojik dersi oldu — **≤0.03 spnte farkına tek seed'le inanma.**

### SNR ölçümü (bu koşunun atlas'ından)

| büyüklük | değer |
|---|---|
| episode / hücre / stage | ~47 |
| medyan \|LP\| | 0.49 |
| medyan `lp_sem` | 0.61 |
| medyan reliability | **0.45** |
| reliability 0.8 için gereken episode | **~23×** |
| ardışık top-10 overlap | **≈ 0.0** |
| TV(uniform) | 0.05 – 0.25 |

Yani LP sinyali kendi ölçüm hatasının altındaydı ve sampler gürültü kovalıyordu.

---

## 3. Ara analiz — kök neden ve offline β replay

Makale ile karşılaştırma net bir teşhis verdi:

- Makalede Fig. 5'te dağılım **sertçe konsantre oluyor.** Bizde olmuyor.
- Steady-state medyan |LP| ≈ 0.5. β = 5 ⇒ `LP/β ≈ 0.1` ⇒ softmax neredeyse düz ⇒ **yapısal uniform** (Cuma).
- Adaptive modda ise reliability × gate çifte küçültmesi aynı sonucu veriyordu (Pazartesi sabah).

Gerçek LP verisiyle **offline β replay**:

| β | ε | sonuç |
|---|---|---|
| 5.0 | 0.02 | ESS ~82 — **ölü** |
| 1.0 | 0.02 | ESS medyan **40–48**, maxp p90 0.15–0.21 — makale rejimi |
| 0.5 | 0.02 | ESS ~10 — **çöküyor** |

Erken 2–3 stage'de |LP| 5–8 çıkıyor, ESS 15–30'a iniyor, ama her stage rebuild + gözlenmemiş hücrelere `LP = 0` imputation ile kendini düzeltiyor (makalede de "önce kolay hücreler" bir özellik, bug değil).

**KARAR:** kod değil, sadece konfig değişikliği →
`temperature_mode = "fixed"`, `β = 1.0`, `ε = 0.02`, `max_cell_probability` 0.08 → 0.25'e gevşetilsin, adaptive kapatılsın.

**Başarı kriteri (eval'den önce, atlas'tan okunacak):**
1. stage ESS **25–60** bandında
2. ardışık top-10 overlap **> 0.3**
3. frontier heatmap'i koherent

---

## 4. Koşu #3 — 27 Temmuz 15:07, `fixed β = 1` → **ÇÖKÜŞ**

**Atlas:** `505449_20260727_150710` · **Model dizini:** `Jul27_15-08-12_v5_lp_acrl_genesis_seed1`

### Konfig (kağıt üzerinde doğru)

```python
V5_BETA                      = 1.0
V5_EPSILON                   = 0.02
V5_TEMPERATURE_MODE          = "fixed"
V5_MAX_CELL_PROBABILITY      = 0.25
V5_MIN_STAGE_EPISODES_FOR_LP = 16
```

### Ne oldu

| stg | step | ESS | maxp | gözlenen hücre | eligible | perf(n) |
|---|---|---|---|---|---|---|
| 2 | 4000 | 11.35 | 0.163 | 84 | 0% | 3.17 (84) |
| 4 | 8000 | **2.34** | **0.645** | 82 | 0% | 7.01 (82) |
| 7 | 14001 | 2.90 | 0.565 | 78 | 0% | 6.97 (78) |
| 9 | 18001 | 3.03 | 0.556 | 66 | 0% | 8.37 (66) |
| 12 | 24001 | **1.24** | **0.895** | 62 | 0% | 7.86 (62) |
| 14 | 28001 | 2.70 | 0.436 | 60 | 0% | 10.12 (60) |
| 16 | 32001 | 1.29 | 0.876 | **54** | 0% | 11.41 (54) |
| 19 | 38002 | 2.33 | 0.555 | 72 | 0% | 10.09 (72) |
| 21 | 42002 | 2.81 | 0.476 | 60 | 0% | 9.22 (60) |

ESS medyanı **2.81**. maxp medyanı **0.56**. TV(uniform) 0.72–0.95. Gözlenen hücre 84 → 54.

Somut olaylar:
- stage 4: `slope_up·L1 @ 1.5–2 m/s` — önceki stage'de **2 episode**, LP = **6.42**, p = **0.645**; o stage'de 1569 episode aldı.
- stage 12: `rough·L4` — önceki stage'de **1 episode**, LP = **9.02**, p = **0.895**.

Dikkat: `eligible_fraction` **her stage'de 0.00**, `max_cell_probability` cap'in (0.25) **üç katına** çıkıyor. İkisi de olamamalıydı.

### Kaçak geri besleme döngüsü

β = 1'i, uniform rejimde ölçülmüş medyan |LP| ≈ 0.49 üzerinden kalibre etmiştim. Ama:

> |LP| gürültüsü **1/√N** ile büyür.

Konsantrasyon başlar başlamaz aç kalan hücrelerde N = 1–2 olur, |LP| 6–9'a fırlar, `e⁹ ≈ 8000` — o hücre dağılımın ~%90'ını alır, flood olur, ortalaması gerçeğe yaklaşır (yani düşer), ve kütle **bir sonraki 1-episode'luk gürültü hücresine** atlar. Döngü kendini besler.

**Offline β replay bunu neden göremedi:** logged LP verisi uniform rejimden geliyordu. Softmax'ın girdi dağılımı, **kendi ürettiği politika altında durağan değil.** Bu, replay yönteminin yapısal sınırı — kalibrasyonu ürettiği rejimin dışına ekstrapole edemez.

---

## 5. Bug avı — iki üretim bug'ı + bir dashboard bug'ı

Çöküş sadece kalibrasyon hatası değildi. `_FiniteEpisodeCurriculum.advance()` içinde iki koruma mekanizması **fixed modda hiç çalışmıyordu.**

### Bug 1 — cap bypass

```python
# ÖNCE (hatalı): cap yalnız adaptive dalda uygulanıyor
weights = (
    self._distribution(scores, self._effective_beta)          # ← _cap_probabilities burada
    if self.temperature_mode == "adaptive_ess"
    else self._mix_epsilon(self._softmax(scores, self._effective_beta))   # ← cap YOK
)
```

`_cap_probabilities` yalnızca `_distribution` üzerinden erişilebiliyordu, o da yalnızca adaptive dalda çağrılıyordu. Sonuç: **`V5_MAX_CELL_PROBABILITY = 0.25` gönderilen freeze'de ölü konfigdi.** maxp 0.895'e çıktı.

```python
# SONRA (düzeltilmiş)
# The per-cell cap is a safety bound on the sampler, not on one
# temperature controller; both modes go through _distribution.
weights = self._distribution(scores, self._effective_beta)
if self.temperature_mode == "adaptive_ess":
    weights = self._ensure_minimum_ess(weights, self._target_ess)
```

### Bug 2 — episode gate bypass

`min_stage_episodes_for_lp` eligibility'si yalnız `_adaptive_scores()` içinde kuruluyordu. Fixed modda **1 episode'luk LP tam ağırlıkla softmax'a giriyordu**, ve `_eligible_masks` hiç güncellenmediği için `eligible_fraction` teşhisi de anlamsız 0.00 okuyordu.

Üç yerde tekrarlanan kapı mantığı tek helper'a çıkarıldı:

```python
def _episode_gate(
    self, progress_mask: np.ndarray, current_sems: np.ndarray
) -> np.ndarray:
    """Cells whose LP rests on enough episodes in BOTH stages to be real."""
    return (
        progress_mask
        & (self._stage_episode_counts >= self.min_stage_episodes_for_lp)
        & (self._previous_stage_episode_counts >= self.min_stage_episodes_for_lp)
        & np.isfinite(current_sems)
        & np.isfinite(self._current_return_sems)
    )
```

ve fixed dal da artık onu kullanıyor:

```python
else:
    # The episode gate is not an adaptive-mode extra.  An LP built
    # from a one-episode mean is noise whose magnitude (|LP| ~ 6-9
    # here) dwarfs the steady-state signal (~0.5), and a bare
    # softmax at beta=1 turns e^{noise} into a runaway: the cell
    # with the loudest measurement error takes ~90% of the mass,
    # floods, regresses, and hands the mass to the next 1-episode
    # cell.  Gated-out cells fall back to the same neutral LP = 0
    # imputation unobserved cells get, so they keep the reference
    # weight e^0 rather than hijacking the distribution.
    self._eligible_masks = self._episode_gate(progress_mask, current_sems)
    scores = self._score(np.where(self._eligible_masks, progress, 0.0))
    self._effective_beta = self.beta
    self._target_ess = float(self._n)
    self._signal_quality = 0.0
```

Ayrıca `sampled_lp_mass` artık yalnız kapıyı geçen hücrelerin LP'si üzerinden hesaplanıyor — aksi halde metrik tek-episode gürültüsüyle dolup okunamaz hale geliyordu.

### Bug 3 — dashboard sessizce frame düşürüyordu

NaN teşhisler `json.dumps` ile standart-dışı `NaN` token'ı üretiyordu → Node `JSON.parse` HTTP 400 dönüyordu → worker 4xx'i **sonsuz retry'lıyor** ve **sonraki bütün frame'leri sessizce düşürüyordu.**

Çözüm `lpacr/dashboard/plugger.py`: özyinelemeli `_json_safe()` sanitizer, `json.dumps(..., allow_nan=False)`, ve 4xx'te retry yerine **break**.

Bu bulunmadan önce bir Opus alt-ajanı soruna checkpoint'ten replay eden bir **sidecar** ile yaklaşmıştı (`dashboard_ckpt_sidecar.py`). Sidecar `build_ued_teacher(cfg)` ile sıfırdan curriculum kurup `state_dict` yüklüyordu; ama `_prev_stage_probabilities` ve teşhis alanları checkpoint'lenmediği için `diagnostics()` `__init__` varsayılanlarını döndürüyordu → dashboard'daki kalıcı `null` metriklerin gerçek kaynağı buydu. Sidecar silindi.

### Bug 4 (kozmetik) — ölü sentinel'ler canlı değer gibi okunuyordu

Fixed modda `target_ess` ve `signal_quality` kullanılmıyor ama state'te sonlu kalmak zorundalar (checkpoint doğrulayıcısı `1.0 <= target_ess <= N` şart koşuyor — `float("nan")`'ı state'e yazma denemesi tam oradan patladı). Dashboard'da "target ESS 84" okunuyordu, yani "kaçmaya çalıştığımız uniform'u hedefliyoruz" gibi. Raporlama sınırında bastırıldı:

```python
adaptive = self.temperature_mode == "adaptive_ess"
...
"target_ess":     float(self._target_ess)     if adaptive else float("nan"),
"signal_quality": float(self._signal_quality) if adaptive else float("nan"),
```

### Düzeltmenin doğrulanması

**Saf gürültü stres testi** (gerçek sınıfla, β=1, ε=0.02):

| | ESS | maxp | kapsama | eligible |
|---|---|---|---|---|
| düzeltilmiş | 12 – 47 | ≤ 0.25 | 83–84/84 | 33–58 |
| eski davranış | 1.05 | 0.97 | 49/84 | — |

**Cap probe** (β=1, ε=0.02, cap=0.25) — yapısal alt sınırlar:

| k sıcak hücre | ESS | maxp | soğuk hücre episode/stage |
|---|---|---|---|
| 1 | 14.43 | 0.250 | 36 |
| 2 | 7.81 | 0.250 | 24 |
| 3 | 5.31 | 0.250 | 12 |
| 4 | 4.16 | 0.245 | 1 ← ε tabanında |
| 8 | 8.30 | 0.123 | 1 |
| 20 | 20.62 | 0.049 | 1 |

ESS tabanı **4.16**; tek hücre devralması artık matematiksel olarak imkânsız.

**Metodolojik ders:** saf gürültü altında `top10_overlap ≈ 0` **doğru** davranıştır (persist edecek gerçek frontier yok). Overlap'i yalnızca gerçek sinyalli koşuda kapı olarak kullan.

**Test:** tam suite **376 passed**, 0 fail. Gate'i özellikle sınayan iki yeni test eklendi (`test_fixed_mode_gates_lp_measured_from_too_few_episodes`, `test_fixed_mode_enforces_the_per_cell_cap`).

---

## 6. Koşu #4 — 27 Temmuz 16:00, düzeltilmiş `fixed β = 1` (**devam ediyor**)

**Atlas:** `505449_20260727_155936` · **Model dizini:** `Jul27_16-00-38_v5_lp_acrl_genesis_seed1`
Konfig koşu #3 ile **birebir aynı** — değişen tek şey kod.

### Sampler seyri (16 stage)

| stg | step | ESS | maxp | elig% | ovl | rel | lpmass | TV | perf |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2000 | 84.00 | 0.012 | 0% | n/a | n/a | n/a | 0.00 | 0.09 |
| 2 | 4000 | 11.06 | 0.174 | 100% | 0.00 | 0.92 | 0.0235 | 0.71 | 4.13 |
| 3 | 6000 | 19.90 | 0.142 | 29% | 0.20 | 0.80 | 0.0430 | 0.57 | 7.00 |
| 4 | 8000 | 74.00 | 0.036 | 29% | 0.10 | 0.47 | 0.0227 | 0.10 | 6.88 |
| 5 | 10000 | 55.03 | 0.079 | 35% | 0.30 | 0.44 | 0.0305 | 0.17 | 7.14 |
| 6 | 12001 | 57.48 | 0.065 | 99% | 0.10 | 0.42 | 0.0214 | 0.18 | 7.36 |
| 7 | 14001 | 26.71 | 0.148 | 99% | 0.00 | 0.40 | 0.0424 | 0.30 | 7.47 |
| 8 | 16001 | 24.18 | 0.152 | 96% | 0.00 | 0.59 | 0.0617 | 0.36 | 7.09 |
| 9 | 18001 | 34.64 | 0.074 | 90% | 0.20 | 0.62 | 0.0331 | 0.40 | 7.27 |
| 10 | 20001 | 58.38 | 0.045 | 74% | 0.10 | 0.54 | 0.0216 | 0.25 | 7.36 |
| 11 | 22001 | 54.19 | 0.043 | 73% | 0.00 | 0.47 | 0.0248 | 0.26 | 7.51 |
| 12 | 24001 | 60.03 | 0.045 | 86% | 0.00 | 0.53 | 0.0201 | 0.23 | 7.89 |
| 13 | 26001 | 65.62 | 0.038 | 87% | 0.00 | 0.39 | 0.0220 | 0.18 | 7.86 |
| 14 | 28001 | 39.55 | 0.102 | 88% | 0.00 | 0.40 | 0.0346 | 0.23 | 7.94 |
| 15 | 30001 | 63.87 | 0.046 | 90% | 0.10 | 0.39 | 0.0206 | 0.19 | 8.15 |
| 16 | 32001 | 62.27 | 0.049 | 93% | 0.00 | 0.46 | 0.0200 | 0.21 | 8.38 |

**Çöküş yok.** ESS en düşük 11.06; maxp en yüksek 0.174 — cap (0.25) hiç devreye bile girmedi. Gözlenen hücre her stage'de 80–84.

**Kendini düzeltme mekanizması gözlemlendi.** Stage 2–3'te dağılım daraldı (TV 0.71) → hücreler açlıktan öldü (medyan 3 episode, 60 hücre kapı altında, eligible %100 → %29) → kapıya takılanlar nötr `LP = 0`'a düştü → stage 4'te dağılım açıldı (TV 0.10, ESS 74) → episode'lar doldu (medyan 44) → eligible %99'a döndü. Yani **salınım bandı, monoton düşüş değil.** Bu, `V5_STAGE_LENGTH_CONTROL_STEPS`'i 2000 → 4000 çıkarma yedek planını gereksiz kıldı.

### Hız ekseninde gerçek bir curriculum

vx marjinali (uniform = 0.25 her biri):

| stg | 0.2–0.5 | 0.5–1 | 1–1.5 | 1.5–2 |
|---|---|---|---|---|
| 2 | **0.943** | 0.033 | 0.013 | 0.011 |
| 3 | **0.647** | 0.247 | 0.053 | 0.053 |
| 4 | 0.319 | 0.220 | 0.230 | 0.230 |
| 5 | 0.367 | 0.208 | 0.220 | 0.206 |
| 6 | 0.344 | 0.227 | 0.220 | 0.210 |
| 7 | **0.487** | 0.185 | 0.164 | 0.163 |
| 8 | 0.189 | **0.524** | 0.144 | 0.143 |
| 9 | 0.263 | **0.524** | 0.144 | 0.069 |
| 10 | 0.236 | 0.334 | 0.303 | 0.128 |
| 11 | 0.164 | 0.316 | **0.355** | 0.165 |
| 12 | 0.206 | 0.290 | 0.281 | 0.223 |
| 13 | 0.213 | 0.227 | 0.302 | 0.257 |
| 14 | 0.169 | 0.301 | 0.269 | 0.261 |
| 15 | 0.242 | 0.255 | 0.297 | 0.206 |
| 16 | 0.220 | 0.301 | 0.215 | 0.264 |

**Yavaş → orta → hızlı, monoton bir süpürme.** 11 stage boyunca hayatta kalan tek tutarlı sinyal budur.

Zorluk seviyesi marjinali de destekliyor: stage 2'de L1 ağırlıklı (0.519), sonra L3/L4'e kayıyor — makaledeki "önce kolay hücreler" davranışı.

### Terrain ekseni: yapı var, yön yok

KL(p ‖ uniform) tam ayrıştırması (nat) — `toplam = KL(vx marjinali) + KL(terrain marjinali) + karşılıklı bilgi`:

| stg | toplam | vx | terrain | etkileşim | vx payı |
|---|---|---|---|---|---|
| 2 | 1.576 | 1.112 | 0.454 | 0.010 | 71% |
| 3 | 0.914 | 0.449 | 0.244 | 0.221 | 49% |
| 4 | 0.056 | 0.012 | 0.014 | 0.030 | 22% |
| 7 | 0.437 | 0.131 | 0.170 | 0.136 | 30% |
| 8 | 0.539 | 0.176 | 0.205 | 0.158 | 33% |
| 9 | 0.495 | 0.233 | 0.150 | 0.113 | 47% |
| 12 | 0.168 | 0.011 | 0.046 | 0.112 | 6% |

Terrain marjinali vx ile **aynı büyüklükte** yapı taşıyor — yani "rastgele" değil. Ama vx'te olan bir şey terrain'de yok: **yön.** Terrain ailesi salınıyor (rough stg 2/5/8'de tepede, slope_down stg 9'da, stairs_down stg 7/12'de), trend yok. `top10_overlap` da bunu doğruluyor.

Bu rotasyonun gerçek sinyal mi gürültü mü olduğu **bu veriden ayırt edilemiyor.** `lp_reliability_median` 0.39–0.62, yani hücre başına LP kabaca kendi SEM'i seviyesinde — büyük ölçüde gürültü yorumuyla uyumlu.

### Koşu #3 ile adil karşılaştırma

Ham ortalamalar karşılaştırılamaz: koşu #3'ün `perf` değerleri, kendi seçtiği **daralan altkümenin** ortalaması (step 24001'de 62 hücre). Aynı hücreler üzerinden:

| step | ortak hücre | koşu #4 | koşu #3 | fark |
|---|---|---|---|---|
| 4000 | 84 | 4.13 | 3.17 | +0.96 |
| 8000 | 82 | 6.89 | 7.01 | −0.12 |
| 14001 | 78 | 7.35 | 6.97 | +0.38 |
| 18001 | 66 | 7.60 | 8.37 | −0.77 |
| 24001 | 62 | 8.03 | 7.86 | +0.17 |

**Eğitim ödülünde tutarlı bir kazanç yok.** Kazanılan şey performans değil, **ölçümün geçerliliği**: koşu #4 her stage'de 84/84 hücreyi ölçüyor, koşu #3 62'ye düşmüştü.

---

## 7. Bütün koşuların özeti

| # | tarih | mod | β | ε | cap | gate | ESS (medyan) | maxp | sonuç |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 24 Tem | fixed | 5.0 | 0.0 | — | — | 82.27 | 0.03 | fiilen uniform; "avantaj" tek-seed varyansıydı |
| 2 | 27 Tem sabah | adaptive_ess | 2.5→(0.92–3.47) | 0.03 | 0.08 | 16 | ~74 | 0.06 | fiilen uniform; holdout **berabere** (0.1295 / 0.1306) |
| 3 | 27 Tem 15:07 | fixed | 1.0 | 0.02 | 0.25 (**ölü**) | 16 (**ölü**) | 2.81 | 0.895 | **çöküş**; 84→54 hücre |
| 4 | 27 Tem 16:00 | fixed | 1.0 | 0.02 | 0.25 | 16 | 59.3 | 0.174 | sampler sağlıklı ama kümülatif maruziyet uniform; holdout **0.1608** — üç koşunun **en kötüsü** (§10) |

### Elimizdeki tek gerçek holdout sonucu

| kol | iter | spnte | succ | cellsucc | fall | CVaR |
|---|---|---|---|---|---|---|
| LP-ACRL (adaptive) | 3500 | 0.1295 | 0.8889 | 0.7619 | 0.0804 | 0.4639 |
| uniform | 2800 | 0.1306 | 0.8919 | 0.7262 | 0.0823 | 0.4655 |

---

## 8. Ne biliyoruz, ne bilmiyoruz

### Kesin olan

1. **LP-ACRL'in uniform'a üstünlüğü henüz hiç gösterilmedi.** Tek holdout ölçümümüz berabere. Cuma'daki avantaj ortadan kalktı.
2. **İlk üç koşuda LP-ACRL fiilen hiç çalışmadı** — ya sabit uniform (β=5, adaptive) ya da çöküş (β=1 buglı). Koşu #4, curriculum'un gerçekten çalıştığı **ilk** koşu.
3. **Kalibrasyon hatam belgelendi.** β=1 uniform rejimde ölçülen |LP| ≈ 0.49 üzerinden türetilmişti; |LP| gürültüsü 1/√N ile büyüdüğü için bu ekstrapolasyon konsantre rejimde geçersiz. `gate = 16` de aynı karakterde — savunulabilir ama bu rejimde doğrulanmadı.
4. **Cap ve gate artık gerçekten devrede**, ve stres testi + cap probe ile ESS tabanı 4.16 olarak sınırlandı.

### Karşılanmayan kriter

Ön-kayıtlı üç kriterden biri **tutmadı**: ardışık top-10 overlap **> 0.3** olmalıydı, 15 fırsatta hiç geçilmedi (maks 0.30). **Kalıcı hücre-seviyesi frontier yok.**

### En kritik olumsuz bulgu

Stage 4'ten sonra **KL(p ‖ uniform) = 0.06 – 0.54 nat**, mümkün olan maksimum `log 84 = 4.43`. ESS 24–66/84. TV 0.18–0.40.

**Örnekleyici bir frontier'dan çok uniform'a benziyor.** Round-2 ile aynı sonuç, farklı sebeple. Bu, "LP ile uniformun arasını açmak" hedefini doğrudan tehdit ediyor.

### Ve bunun neden bir bug değil, bir takas olduğu

Gate, düşük-N hücreleri elemek zorunda. Elenen hücre nötr `LP = 0`'a düşer. `LP = 0` softmax'ta `e⁰` = referans ağırlık demek — yani **matematiksel olarak uniform'a doğru çekiş.**

> Bizi çökmekten kurtaran mekanizma ile bizi uniform'a geri iten mekanizma **aynı** mekanizmadır.

β tek kaldıraç ve üç ayarı da denendi:

| β | gate | sonuç |
|---|---|---|
| 5 | — | uniform |
| 1 | yok | çöküş |
| 1 | var | hafif konsantrasyon (şu an) |

Ortada dördüncü bir ayar yok.

### Mevcut tasarımla umutsuz olan

84 hücrelik uzayda, stage başına ~40 episode/hücre ile **büyük** bir LP-uniform farkı çıkarmak. Aritmetik tutmuyor: reliability 0.8 için ~23× episode gerekiyor, ve stage'i o kadar uzatırsak curriculum adapte olamaz.

### Umutsuz olmayan

vx progresyonu. 15 stage boyunca hayatta kalan tek tutarlı sinyal o, ve **bir vx bandı içinde 21 terrain hücresini havuzlamak tahmin başına ~21× episode demek** — ihtiyaç duyulan 23×'in tam mertebesi.

Yani veri, daha önce sadece hipotez olan "uzayı küçültme, **yapı-farkındalıklı varyans azaltma** (spatial pooling / faktörize 2-seviye örnekleme + kayan pencere regresyon LP)" yönünü ilk kez rakamla destekliyor.

---

## 9. Sıradaki adımlar

1. Koşu #4'ü 3500 iterasyona kadar bitir (ETA ~50 dk).
2. Validation seçimi + **holdout final** eval. Karar buradan çıkacak.
3. Beklenti (dürüst): ESS ~55 ve KL ~0.2 ile farkın yine küçük çıkmasını bekliyorum. Küçük çıkarsa bu bir başarısızlık değil — pooling kararı için ihtiyacımız olan kanıt olur.
4. Kapılar geçerse: seed 17 ve 23 ile çok-seed kampanya. **Tek seed'e asla inanma** (bkz. bölüm 1).
5. `diagnostics()` NaN düzeltmesi koşu bitince kümeye senkronize edilecek (koşan süreç modülünü zaten belleğe aldı; bu koşu boyunca panelde `target ESS 84` görünmeye devam edecek).

---

## 10. Koşu #4 tamamlandı — 41 stage'in tam analizi (28 Temmuz sabahı)

Koşu 27 Tem 17:27'de **temiz bitti**: 3500 iterasyon, 344M timestep, 5160 s
(86 dk, 1.46 s/iter), `model_3500.pt` yazıldı, log'da tek uyarı Genesis'in
bilinen self-collision filtresi. Atlas 41 frame (step 2000 → 82002 ≈ iter 3417).

**Validation seçimi ve holdout eval'i HENÜZ ÇALIŞMADI.** §9'un 2. adımı açık:
interaktif tahsis (job 505449) 28 Tem 04:23'te düştü, kümede koşan iş yok.
Aşağıdaki her şey eğitim-içi ölçüm; kararı verecek sayı hâlâ eksik.

### 10.1 Sampler sağlığı: sorun yok

| | stage 2–3 | stage 4–16 | stage 17–41 |
|---|---|---|---|
| KL(p ‖ uniform) | 1.24 | 0.237 | **0.169** |
| ESS medyanı | 15.5 | 55.0 | **61.2** (35.8–70.1) |
| maxp | 0.174 / 0.142 | ≤0.152 | ≤0.113 |
| eligible% | 100 / 29 | 29→93 | **0.89–0.98** |

Cap (0.25) **hiç bağlamadı**; ESS tabanı 11.06. Kapsama 84/84 — tek istisna
stage 3'te 80 (stage 2'nin daralmasından 4 hücre aç kaldı, stage 4'te geri geldi).
Koşu #3'ün çöküşü tekrarlamadı, düzeltmeler tuttu. Steady-state'te gate de
neredeyse hiç bağlamıyor (elig ≈ 0.92) — yani §8'deki "bizi uniform'a iten şey
gate'tir" açıklaması koşu #4 için **yanlış**; asıl sebep aşağıda.

### 10.2 Kümülatif maruziyet fiilen uniform

Stage başına KL ~0.17 nat gürültüsü **ortalamada sıfırlanıyor**:

| ölçü | değer | uniform |
|---|---|---|
| stage-ortalama p, KL | **0.0117 nat** (maks 4.43) | 0 |
| gerçekleşen episode payı, KL | **0.0068 nat** | 0 |
| en çok / en az örneklenen hücre oranı | **1.65×** | 1.0 |
| toplam episode | 204 453 (≈50/hücre/stage) | — |

3500 iterasyon boyunca LP-ACRL'in politikaya sunduğu eğitim diyeti, hücre
başına ±%25 içinde **uniform'un aynısı.** Tek gerçek sapma stage 2–3 (KL 1.58 /
0.91), yani eğitimin ilk %7'si.

### 10.3 Neden: LP tahmini kendi gürültüsünün altında

Ardışık stage'lerin LP vektörleri arasındaki korelasyon:

```
corr(LP_t, LP_{t+1})  medyan = −0.446      (37 stage çifti)
saf gürültü teorik siniri     = −0.500
```

Bu tesadüf değil, **kapalı form**. `LP_t = perf_t − perf_{t−1}` olduğu için
ardışık iki LP `perf_t` terimini paylaşır; gerçek sinyal yoksa korelasyon tam
`−σ²/2σ² = −0.5` olur. Gerçek bir sürüklenme `d` varsa korelasyon 0'a doğru
itilir. Ölçülen −0.446 ⇒

- sinyal/gürültü **varyans** oranı ≈ 0.24
- sinyal/gürültü **genlik** oranı ≈ **0.49**

Bağımsız doğrulama: `|LP|` medyanı 0.39, tek-stage `perf_sem` medyanı 0.36 →
farkın gürültü tabanı `√2 × 0.36 = 0.51`. **LP'nin tipik büyüklüğü kendi
gürültü tabanının altında.** `lp_reliability_median ≈ 0.40` de bunu söylüyor.

Sonuç: sampler her stage'de p_max/p_min = 10–100× oranlar üretiyor (LP yayılımı
2.5–4.8, β=1), ama bunu **her seferinde başka hücrelere** veriyor. Ardışık
sapma vektörlerinin kosinüs benzerliği −0.24 (uzak-stage null: +0.00 ± 0.13) —
sadece "kalıcılık yok" değil, sistematik **ters dönme** var; bu da fark
tahmincisinin mean-reversion imzası. Örnekleme→LP geri beslemesi ise temiz:
`corr(episode_sayısı_t, LP_{t+1}) = +0.02`, yani cap+gate hijyeni çalışıyor,
sampler sadece simetrik gürültü kovalıyor.

`top10_overlap` 40 fırsatta medyan 0.00, maks 0.30 → ön-kayıtlı **>0.3 kriteri
41 stage'de de tutmadı.**

### 10.4 vx sinyali erken bir geçici olaymış

§6'da "11 stage boyunca hayatta kalan tek tutarlı sinyal" denen yavaş→hızlı
süpürme, tam koşuda **stage 17'de ölüyor**:

| | vx ağırlık merkezi ~ stage korelasyonu | com aralığı |
|---|---|---|
| stage 2–16 | **r = +0.76** | 0.09 → 1.52 |
| stage 17–41 | r = +0.28 | 1.22–1.59 (uniform = 1.50, ort 1.44) |

Yani gerçek curriculum sinyali eğitimin ilk ~1400 iterasyonunda (politika henüz
yavaş komutlarda bile beceriksizken, hücreler arası gerçek perf farkı gürültüyü
aşarken) vardı. Politika geniş yeterliliğe ulaşınca hücreler arası fark SEM'in
altına indi ve LP kalıcı olarak gürültüye döndü. Terrain ekseninde hiçbir
aşamada yön yoktu (aile marjinalleri 41 stage boyunca 0.19 etrafında salınıyor).

### 10.5 Uniform ile karşılaştırma (eğitim-içi, tek seed)

Karşılaştırılan kol: `Jul27_11-01-19_v5_uniform_genesis_seed1` (+ önceki
segmentleri). Kaydedilen `go2_v5_config.py`'lerin diff'i **yalnız LP'ye özgü
knob'ları** içeriyor (β, ε, mod, cap, gate) — fizik/ödül/env birebir aynı,
ikisi de seed 1, 4096 env, 3500 iter. Atlas `step` = `global_control_steps`
checkpoint'e yazıldığı için resume'lu uniform kolu ile step hizası doğru.

Ortak hücreler üzerinden hücre-başına ortalama episode getirisi:

| iter ≈ | koşu #4 (LP) | uniform | fark |
|---|---|---|---|
| 250 | 7.00 | 5.57 | **+1.43** |
| 1250 | 8.15 | 10.50 | −2.36 |
| 1500 | 8.59 | 10.98 | −2.39 |
| 2000 | 9.03 | 10.34 | −1.30 |
| 3416 | 9.16 | 10.28 | **−1.12** |

TB `Train/mean_reward` (50-iter penceresi) aynı yönü veriyor: iter 3499'da
LP #4 **12.29**, uniform **12.80**, koşu #2'nin LP kolu 13.04. Episode uzunluğu
eşit (925 / 927).

Yani düzeltilmiş LP-ACRL, curriculum'un gerçekten çalıştığı ilk koşuda
uniform'un **önüne geçmedi; ~1 puan gerisinde bitirdi.** Tek seed, ve
kümülatif diyet uniform'la aynı olduğu için bu farkın büyük kısmı muhtemelen
seed-içi varyans — ama **hiçbir okumada LP lehine bir işaret yok.**

### 10.6 Ne değişti, ne değişmedi

Değişen: koşu #4 sampler'ın **sağlıklı** çalıştığı ilk koşu. Cap, gate,
kapsama, kendini düzeltme — hepsi çalışıyor. Ölçüm artık geçerli.

Değişmeyen: §8'in ana bulgusu. Üç farklı rejim (β=5, adaptive, düzeltilmiş
β=1) **aynı yere** çıktı — kümülatif olarak uniform. Sebep artık net ve
sayısal: 84 hücre × ~50 episode/stage ile LP'nin genliği kendi gürültüsünün
yarısı. β bunu düzeltemez, çünkü β gürültüyü de sinyal kadar büyütüyor
(koşu #3 tam olarak buydu).

Bu, §8'in "umutsuz olmayan" yönünü doğrudan destekliyor: **bir vx bandı içinde
21 terrain hücresini havuzlamak** tahmin başına ~21× episode demek, ve
genlik/gürültü oranını 0.49 → ~2.2'ye çıkarır. Sinyalin var olduğu tek eksen
zaten vx (§10.4).

### 10.7 Sıradaki adım

1. `scripts/run_v5_selection.sh` + holdout eval'i `Jul27_16-00-38` için çalıştır
   (yeni tahsis gerek). Beklenti: uniform ile berabere ya da hafif geride.
2. Sonuç ne olursa olsun karar aynı: tek hücreli LP tahmini bu bütçede
   ölçülemez. Faktörize (vx-bandı × terrain) 2-seviye örnekleme + kayan pencere
   regresyon LP'ye geç.
3. Tek seed'e inanma — ama bu koşuda inanılacak bir avantaj da yok.

### 10.8 Validation + holdout sonucu (28 Tem, job 508168)

`v5select` job 508168, a100q/a122, 18:34, rc=0. Holdout ayağı bu sefer normal
akışta çalıştı — 27 Tem'deki `rc=2` workaround'una gerek kalmadı.
Artifact kökü: `logs/v5_selection/508168_20260728_083456/`.

Üç koşunun tamamı **aynı validation bank** (`dd1a2cd0…`), aynı holdout bank
(`a00aa2ed…`) ve aynı `protocol_fingerprint` (`db3541a8…`) ile ölçüldü;
karşılaştırma apples-to-apples. Hepsi seed 1, 4096 env, 3500 iter; kaydedilmiş
`go2_v5_config.py` diff'i yalnız LP knob'larını içeriyor.

Validation eğrisi (macro_mean_spnte_lin, **düşük = iyi**):

| iter | LP #4 (fixed β=1) | LP #2 (adaptive) | uniform |
|---|---|---|---|
| 1000 | 0.2412 | — | — |
| 1400 | 0.2065 | — | — |
| 1800 | 0.1904 | — | — |
| 2200 | 0.1766 | — | — |
| 2400 | — | 0.1465 | 0.1410 |
| **2600** | **0.1584** ← seçildi | — | — |
| 2800 | — | 0.1568 | **0.1387** ← seçildi |
| 3000 | 0.1723 | — | — |
| 3200 | — | 0.1423 | 0.1522 |
| 3500 | 0.1795 | **0.1393** ← seçildi | 0.1395 |

cell_success_rate aynı noktalarda: LP #4 en iyi 0.6786 (iter 2600/3000) —
uniform 0.8095, LP #2 0.7976. fall_rate LP #4'te iter 3500'de 0.1121'e çıkıyor.

Holdout final:

| koşu | seçilen iter | **spnte** | succ | cellsucc | fall | CVaR |
|---|---|---|---|---|---|---|
| **LP #4 (fixed β=1)** | 2600 | **0.1608** | 0.8661 | 0.6548 | 0.0823 | **0.4393** |
| LP #2 (adaptive) | 3500 | 0.1295 | 0.8889 | 0.7619 | 0.0804 | 0.4639 |
| uniform | 2800 | 0.1306 | 0.8919 | 0.7262 | 0.0823 | 0.4655 |

**Curriculum'un gerçekten çalıştığı ilk koşu, elimizdeki en kötü koşu.**
Fark 0.030 — tek başına "≤0.03'e inanma" eşiğinde. Ama bu dört bağımsız
okumanın dördüncüsü ve hepsi aynı yöne bakıyor:

| okuma | LP #4 − uniform |
|---|---|
| TB `Train/mean_reward`, iter 3499 | −0.51 |
| atlas ortak-hücre perf, iter ~3416 | −1.12 |
| validation spnte, iter 3500 | +0.0400 (kötü) |
| holdout spnte, seçilen ckpt | +0.0302 (kötü) |

Tek seed hâlâ tek seed; ama "berabere" bile diyemiyoruz — bütün işaretler
LP #4 aleyhine. §10.3'ün mekanizmasıyla tutarlı: sampler her stage'de
10–100× oranlar üretip bunları gürültüye göre dağıtıyor, kümülatif diyet
uniform'la aynı kalıyor, geriye sadece **fazladan varyans** kalıyor.

Tek olumlu ayrıntı: LP #4'ün CVaR'ı üç koşunun en iyisi (0.4393 vs 0.4639 /
0.4655) ve iter 2600'de validation CVaR'ı 0.4365 ile bandın çok altında —
kuyruk riski daha iyi, ortalama daha kötü. Tek nokta, üzerine iddia kurulmaz;
ama pooling tasarımında CVaR'ı ayrı takip etmeye değer.

Ayrıca LP #4'ün validation eğrisi 2600'den sonra **bozuluyor**
(0.1584 → 0.1723 → 0.1795), uniform ise 0.139–0.152 bandında düz kalıyor.
Geç eğitimde gürültü-kovalayan örneklemenin zarar verdiği yorumuyla uyumlu,
ama tek eğriyle kanıtlanmaz.

**Karar değişmiyor:** hücre-seviyesi LP bu bütçede ölçülemiyor (§10.3).
Faktörize (vx-bandı × terrain) 2-seviye örnekleme + kayan pencere regresyon
LP'ye geç; β ile oynamayı bırak.

---

## 11. Analiz turu — 28 Temmuz, atlas üzerinde ölçüm (eğitim yok)

Koşu #4'ün holdout sonucu geldikten sonra (§10.8) yeni koşu yapmadan, yalnız
mevcut atlas üzerinde bir teşhis turu yapıldı. Amaç §10.7'nin "faktörize
havuzlamaya geç" kararını uygulamadan önce, LP'nin neden ölçülemediğini
sayısallaştırmaktı. Sonuç kararı değiştirdi.

Tüm scriptler `lpacr/analysis/` altında ve tekrar çalıştırılabilir:

```bash
.venv/bin/python -m lpacr.analysis.lp_diagnostics      # 11.1 - 11.6
.venv/bin/python lpacr/analysis/sample_size_power.py   # 11.7
```

### 11.0 Metodolojik kural — her teşhis kendi null'ı ile

Bu turun en kalıcı çıktısı bir sayı değil, bir kural. Üç ayrı teşhis
(`lp_reliability`, kesitsel sinyal kapısı, `corr(LP_t, LP_{t+1})`) daha önce
"büyük/küçük" diye okundu; üçünün de null'ı hesaplanmamıştı ve üçü de gürültü
tabanında oturuyordu. Bundan sonra:

> **Null'ı türetilmemiş bir sayı kapı olarak kullanılamaz.**

### 11.1 Stage sansürü ölçüldü: her stage'de completion'ların %46'sı atılıyor

`observe()` bir episode'u LP'ye ancak `assigned_revision == sampler_revision`
ise kabul ediyor (`episode_curriculum.py:291`). Stage 2000 control step,
ortalama episode ~1000 step ⇒ bu nadir bir kenar durumu değil.

| ölçü | değer |
|---|---|
| stage başına atılan completion oranı | **0.463** [0.458, 0.467] |
| toplam atılan episode | **143 523** / 312 568 |
| kabul edilen episode/hücre/stage | 41 — sansürsüz 83 (**2.02×**) |
| hücre başına late-fraction aralığı | 0.299 → 0.521 |
| `corr(late_fraction, performance)` | **+0.705** (null 0.000) |

Sansür rastgele değil: uzun episode sınırı daha çok aşar, uzun episode başarılı
olandır ⇒ **robot nerede iyiyse orada orantısal olarak daha çok veri atılıyor.**
Kabul edilen alt küme düşüşlere doğru eğik, ve bu eğimin büyüklüğü fall/timeout
karışımıyla birlikte stage'ler arası kayıyor — yani `LP = Δ(admitted mean)`
kısmen Δ(sansür kompozisyonu) ölçüyor.

Bias'ın *büyüklüğü* (admitted mean vs all-completions mean) mevcut atlastan
çıkarılamıyor; yeni enstrümantasyonun sansürsüz akümülatörleri bunun için.

### 11.2 Sinyal kapısı α ve yörüngesi

İki bağımsız tahminci:

```
alpha_SEM      = clip( (Var_cells(LP) - E[LP_SE^2]) / Var_cells(LP), 0, 1 )
alpha_temporal = clip( 1 + 2*corr(LP_t, LP_{t+1}), 0, 1 )
```

İkincisinin türetimi kapalı form: `LP_t = P_t - P_{t-1}` ardışık iki LP `P_t`
terimini zıt işaretle paylaştığı için `corr = -sigma_g^2/(sigma_s^2 + 2 sigma_g^2)`,
tersine çevirince `1 + 2corr = sigma_s^2 / Var(LP)`. corr −0.5 ⇒ α=0, corr 0 ⇒ α=1.
**SEM kullanmıyor.**

Stage başına `alpha_SEM` (84 hücre):

```
0.97 0.27 0.02 0.00 0.00 0.33 0.50 0.67 0.30 0.23 0.08 0.14 0.29 0.00 0.00 0.00
0.21 0.00 0.00 0.30 0.07 0.00 0.03 0.00 0.00 0.00 0.04 0.00 0.00 0.00 0.00 0.00
0.00 0.00 0.01 0.00 0.00 0.00 0.00 0.00
```

**40 stage'in 27'sinde α = 0.** Rejim bazında:

| rejim | α_SEM | lag-1 corr | α_temporal |
|---|---|---|---|
| stage 2–16 | 0.228 | −0.337 | 0.327 |
| stage 17–41 | **0.000** | −0.451 | **0.099** |

Çapraz doğrulama: α_SEM'in erken medyanı 0.228, §10.3'ün lag-1 korelasyonundan
türeyen bağımsız tahmin 0.195. İki farklı yoldan aynı büyüklük.

### 11.3 α_SEM'in kör noktası: ortak-mod gürültü (§10.6 havuzlama hipotezine darbe)

§10.6 "bir vx bandı içinde 21 terrain hücresini havuzla ⇒ 21× episode ⇒
genlik/gürültü 0.49 → 2.2" diyordu. Test edildi:

| birim | α_SEM (geç) | lag-1 corr | α_temporal |
|---|---|---|---|
| vx bandı (4 birim, 21× havuz) | **0.960** | −0.448 | 0.103 |
| terrain hücresi (21 birim, 4× havuz) | **0.961** | −0.572 | 0.000 |
| 84 hücre (havuzsuz) | 0.000 | −0.446 | 0.108 |

İkisi aynı anda doğru olamaz: α=0.96 gerçek olsaydı lag-1'in 0'a yakın olması
gerekirdi. Yanılan α_SEM, ve sebebi yapısal:

> `lp_se` hücre-içi episode saçılımından hesaplanıyor, yani yalnız **hücreler
> arası bağımsız** gürültüyü görüyor. Ama 84 hücre tek bir politikayı paylaşıyor;
> bir PPO güncellemesi hepsini birlikte kaydırıyor. Bu **ortak-mod** bileşen
> SEM'e görünmüyor. Havuzlama bağımsız kısmı √21 ile eziyor, ortak kısma
> dokunmuyor ⇒ α_SEM aşırı güvene gidiyor.

**Havuzlama zamansal kalıcılığı hiç iyileştirmiyor** (α_temporal 0.108 → 0.103).
Bu, §10.7'nin faktörize havuzlama kararına birinci uyarı ateşi.

Kapı bu yüzden `alpha = min(alpha_SEM, alpha_temporal)` olmalı: biri ortak-mod
gürültüye kör, diğeri kesitsel ayrışmaya kör.

### 11.4 Kayıtlı `lp_reliability` metriği hiçbir zaman sinyal ölçmemiş

`episode_curriculum.py:399`:

```
reliability = |LP| / (|LP| + lp_sem)
```

Saf gürültü altında `LP ~ N(0, s^2)`, `lp_sem ~ s`, `E|LP| = s*sqrt(2/pi) = 0.798s` ⇒

| | değer |
|---|---|
| teorik saf-gürültü sabit noktası | **0.444** |
| ölçülen medyan, stage 17–41 | **0.414** (fark 0.029) |

Bu metrik **sıfır döndüremiyor**; girdide hiç sinyal olmasa bile ~0.44'e oturuyor.
§2'de "reliability medyanı 0.45" ve §10.3'te "lp_reliability_median ≈ 0.40" diye
raporlanan ve orta düzey güven olarak okunan sayılar, gürültü tabanının
kendisiydi. Geriye dönük bir yanlış okuma düzeltildi.

### 11.5 MAD ile ölçek normalizasyonu tek başına mayın

Farklı LP tanımlarının farklı fiziksel birimleri var (ham return farkı ~0.5,
reward-per-step farkı ~0.0005), bu yüzden β'yı boyutsuzlaştırmak için
`z = LP / MAD(LP)` gerekiyor. Ama MAD **kuyruğu kasıtlı olarak yok sayar**,
softmax ise yalnız kuyrukla çalışır:

| | β=1'de max/min softmax oranı |
|---|---|
| medyan stage | **273×** |
| en kötü stage | **548 905×** |
| `clip(z, ±3)` ile sınırlı | 403× |

Çıplak MAD normalizasyonu koşu #3'ün çöküşünü tekrarlardı. **Ölçek
normalizasyonu zorunlu ama tek başına yeterli değil**; yanına sert z-clip ve
§11.2'nin sinyal kapısı gerekiyor. Skorlama üç katmanlı: ölçek (z + clip),
sertlik (β), kapı (α).

### 11.6 Öngörü testinin null'ı pencere tasarımına bağlı — ve sinyalin ufku ~1 stage

`Score = B − A` ile `Future = C − B` tasarımı **B penceresini zıt işaretle
paylaşıyor**, yani saf gürültü altında −0.5 döndürüyor. Bu tasarım
`corr(LP_t, LP_{t+1})`'in birebir kendisi.

| tasarım | Pearson | Spearman | null | null-düzeltilmiş | permütasyon null |
|---|---|---|---|---|---|
| 3 pencereli (B ortak) | −0.417 | −0.411 | −0.5 | **+0.083** | −0.007 |
| 4 pencereli (tam ayrık) | **+0.042** | **+0.076** | 0.0 | +0.042 | −0.007 |
| 5 pencereli (ayrık, lag 3) | −0.020 | −0.066 | 0.0 | −0.020 | +0.002 |

Ek tuzak: **hücre etiketi permütasyonu bu artefaktı yakalamıyor** (her iki
tasarım için de null ≈ 0.00), çünkü paylaşılan-B etkisi hücre-*içi* bir zaman
eşleşmesi. Doğru null ya analitik ya hücre-içi blok permütasyonu.

**Yeni bulgu — sinyal sıfır değil, ufku kısa.** §10.3'ün "−0.446, saf gürültü
sınırı −0.5, sinyal yok" okuması mekanik artefaktı sinyalin yokluğuyla
karıştırıyordu. Artefakt çıkarılınca lag 1'de +0.083, lag 2'de +0.042, lag 3'te
tükeniyor. Yani **kullanılabilir sinyalin ömrü ~1 stage (~83 iterasyon)** —
sampler ölçüp bir sonraki stage'de tepki verdiğinde sinyal zaten bitmiş.
"Kümülatif diyet neden uniform çıktı" sorusuna §10.3'ten daha iyi cevap:
sinyal yok olduğu için değil, **ömrü tepki gecikmesinden kısa olduğu için.**

Estimator yarışmasının birincil hedefi bu yüzden korelasyonu +0.04'ten +0.08'e
çıkarmak değil, **ufku uzatmak.**

### 11.7 Örnek büyüklüğü analizi: episode sayısı bağlayıcı kısıt DEĞİL

3322 hücre-stage üzerinde `LP_SE = sqrt(sem_t^2 + sem_{t-1}^2)`, `z = |LP|/LP_SE`,
`N_harm = 2/(1/N_t + 1/N_{t-1})`.

**Sinyal varyansı ayrıştırması, geç rejim (2100 satır):**

```
sigma_signal^2 = mean(LP^2) - mean(LP_SE^2) = 0.4205 - 0.4078 = +0.013  ~= 0
```

İki baskın stratumda (N=25–40 ve 40–60, 2100 satırın 1746'sı) **negatif**.
Gözlenen `P(z>1) = 0.329` / `P(z>2) = 0.056`, saf gürültü null'ı 0.3173 / 0.0455
⇒ fazla yalnız **+0.012 / +0.011**.

**k× bütçe ekstrapolasyonu** (kalibre edilmiş spike-and-slab; Gauss moment
yöntemi ağır kuyruk yüzünden k=1'i +0.14 fazla tahmin ettiği için birincil
alınmadı). 78 uygun hücre/stage:

| bütçe | z>2 hücre | null 3.5 üzeri fazla |
|---|---|---|
| 1× (şimdi) | 4.2 | +0.7 |
| **2× (sansür kaldırılınca)** | 4.8 | **+1.2** |
| 4× | 5.8 | +2.2 |
| 8× | 7.3 | **+3.7** |

**8× episode, stage başına ~3 ekstra gerçek sinyal alıyor.** Sansürü kaldırmak
yarım hücre. Sebep: N artınca `LP_SE` küçülüyor ama `|LP|` de aynı oranda
küçülüyor; sinyal yoksa `z` yerinde kalıyor.

**`sem ~ N^b` ölçeklemesi — iki rejimin patolojisi zıt:**

| | erken 2–16 | geç 17–41 | teori |
|---|---|---|---|
| b (iki-yönlü FE) | **−0.332** | **−0.466** | −0.500 |
| varyans tabanı C | SE'nin %90'ı geri alınamaz | %88'i geri alınabilir | — |
| sinyal (excess P(z>2)) | +0.19 | +0.01 | — |

Geç rejimde ölçekleme kusursuz — gürültü gerçekten örnekleme gürültüsü ve
düzgün küçülüyor; **altında ölçülecek bir şey yok.** Erken rejimde sinyal var
ama `1/sqrt(N)` kırık: varyans tabanı 8× bütçenin getirisinin ~%40'ını yiyor.
`LP_SE`'yi yarıya indirmek için gereken N: teoride 168, geçte 186, **erkende 340.**

Erken b'nin yaklaşık yarısı seçim artefaktı: çok örneklenen hücreler aynı
zamanda içsel olarak yüksek varyanslı hücreler (`spearman(N, sd_ep)` = +0.332
erken, +0.034 geç). Hücre sabit etkileriyle −0.220 → −0.332.

**Düşük-N hücreleri ve runaway öncüsü:**

| ölçü | erken | geç | null |
|---|---|---|---|
| stage-içi `spearman(N, \|LP\|)` | −0.144 | −0.178 | 0 |
| stage-içi `spearman(N, p)` | −0.014 (n.s.) | **−0.295** | 0 |
| `spearman(N, z)` | +0.096 | **−0.001 (n.s.)** | 0 |
| top-10 p hücresinin alt-N çeyreğinden gelme oranı | 0.133 | **0.480** | 0.25 |

Geç rejimde en yüksek olasılıklı 10 hücrenin **%48'i en düşük N çeyreğinden**.
Ve `corr(N, z) ≈ 0` — düşük N'deki fazla `|LP|` tam olarak fazla gürültü.

Ama döngü şu an **negatif geri besleme**: `p[t] → N[t+1]` ρ=+0.94 (neredeyse
deterministik), `N[t+1] → |LP|[t+1]` ρ=−0.14. Terfi eden hücre çok episode
alıyor, gürültüsü küçülüyor, `|LP|`'si düşüyor, tenzil ediliyor. **Koşu #4'ün
çökmeme sebebi bu.** β yükseltilir veya gate gevşetilirse koruma kalkar.

**Gate doğru çalışıyor, kalsın.** `min_stage_episodes_for_lp = 16` elediği
hücrelerin `|LP|`'si 2× büyük (0.711 vs 0.369, p=4e-15) ama `z`'si aynı
(0.755 vs 0.700, **p = 0.11**). Sinyal kaybı yok, gürültü eleniyor. Satırların
yalnız %11.9'u eleniyor; yukarıdaki düşük-N amplifikasyonu gate'i **geçen**
hücreler arasında oluyor.

**Anomali — `performance_sem` fazla tahmin ediliyor olabilir.** Geç 25–40
stratumunda gözlenen `P(z>1) = 0.290`, null 0.3173'ün **altında**; `sigma^2`
negatif. Muhtemel açıklama: bir hücre tek bir `(vx_bin, terrain_cell)` ama komut
`vx` o bandın içinde, `vy` ve `omega_z` taban aralıklarından sürekli
örnekleniyor (`genesis_adapter.py:152`, `legged_robot.py:641` — komut episode
ortasında da yeniden çekiliyor). Bu hücre-içi heterojenlik iki stage'de de aynı,
yani **farkta sadeleşiyor**; ama `LP_SE = sqrt(sem_t^2 + sem_{t-1}^2)` onu iki
kez sayıyor.

Doğruysa: **gürültünün bir kısmı episode ile değil, tahminle kaldırılabilir** —
her episode'un getirisini kendi komutuna göre residualize et, aynı veriyle
`LP_SE` düşsün. Enstrümantasyona per-episode komut kovaryatları
(`mean_cmd_vx/vy/yaw`, `cmd_vx_std`) bu yüzden eklendi.

### 11.8 Ne değişti

**§10.7'nin kararı revize edildi.** "Faktörize havuzlama + kayan pencere
regresyon LP'ye geç" planının iki gerekçesinden biri düştü:

- *"21× episode ⇒ SNR 0.49 → 2.2"* — §11.7 diyor ki geç rejimde episode
  katlamak `z`'yi kıpırdatmıyor (8× ⇒ +3.7 hücre/stage). §11.3 diyor ki
  havuzlama zamansal kalıcılığı hiç iyileştirmiyor. **İki bağımsız okuma aynı
  yönde.** Havuzlamaya gitmeden önce bunlar cevaplanmalı.

**Konu sıralaması değişti:**

| konu | önceki beklenti | §11 sonrası |
|---|---|---|
| sansür kaldırma | LP gürültüsünü çözer | **bias düzeltmesi**, güç düzeltmesi değil (+0.5 hücre/stage). Yine de yapılacak: `corr(late, perf) = +0.705` gerçek ve yanlış. |
| reward-per-step | ikincil | **birincil.** Geç rejimde hücre-başı ortalama getiri doymuş; sorun estimator değil metrik olabilir. |
| kovaryat düzeltmesi | yoktu | **yeni aday (F).** Episode gerektirmeyen varyans azaltma. |
| havuzlama | §10.7'nin ana planı | iki uyarı ateşi aldı, önce §11.3/§11.7 cevaplanmalı |

**Ve dürüst ihtimal:** politika geniş yeterliliğe ulaştıktan sonra ölçülecek
learning progress gerçekten kalmamış olabilir. α kapısı bu durumda "uniform
örnekle" der ve **doğru** cevap odur. Bu bir başarısızlık değil, ölçülmüş bir
sonuçtur ve öyle yazılmalıdır.
