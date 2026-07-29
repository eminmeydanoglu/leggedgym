# V5 UED / LP-ACRL — Tüm Denemelerin Geniş Özeti

**Dönem:** 23–28 Temmuz 2026  
**Platform:** Go2 · Genesis · legged_gym + rsl_rl (PPO) · 4096 env · UHeM Altay A100  
**Amaç:** Learning-Progress ACRL (Li/Li/Hutter) sampler’ını V5 UED task space’inde uniform baseline’a karşı göstermek  
**Kaynaklar:** Curriculum Atlas frame’leri (`logs/lpacr_dashboard_data/`), holdout eval (`heldout_final/`), `lpacr/HISTORY.md`  
**Bu belge:** atlas’tan yeniden hesaplanan sayılar + kayıtlı validation/holdout sonuçlarının birleştirilmiş özeti

---

## 1. Tek cümlelik sonuç

**Hücre-seviyesi LP-ACRL, bu bütçede (84 hücre, ~50 episode/hücre/stage, tek seed) uniform’a üstünlük göstermedi.** Dört ana rejim denendi; ikisi fiilen uniform örnekledi, biri çöktü (üretim bug’ı + kalibrasyon), düzeltilmiş “çalışan” rejim ise holdout’ta **üç koşunun en kötüsü** oldu. Kalıcı hücre-seviyesi frontier yok; LP sinyali kendi ölçüm gürültüsünün altında.

---

## 2. Deney kurgusu (tüm koşularda ortak)

### 2.1 Task space

| | |
|---|---|
| Boyut | **84 hücre** = 4 `vx_bin` × 21 `terrain_cell` |
| `vx_bin` | 0.2–0.5 · 0.5–1 · 1–1.5 · 1.5–2 m/s |
| Terrain | stairs_up/down L1–4, slope_up/down L1–4, rough L1–4, flat L1 |
| Fingerprint | `dc108261…` (tüm koşularda aynı) |
| Stage | 2000 control step / stage |
| Seed (başarılı pair’ler) | **1** (tek seed; çok-seed kampanya yapılmadı) |

### 2.2 LP formülasyonu

```
LP_cj(ζ) = R_cj(ζ) − R_c(j−1)(ζ)
p(ζ)     = softmax( LP(ζ) / β )   # stage başında sıfırdan; makalede koruma yok
```

Serbest parametre pratikte **β** (+ sonradan ε, cap, episode gate, temperature mode).

### 2.3 Ana metrikler

| Metrik | Yön | Anlam |
|--------|-----|--------|
| `macro_mean_spnte_lin` | **↓ iyi** | Holdout/validation birincil skor |
| `macro_success_rate` | ↑ | Replika başarı |
| `cell_success_rate` | ↑ | 84 hücreden kaçında başarı |
| `macro_fall_rate` | ↓ | Düşme |
| `worst_10pct_cvar_spnte_lin` | ↓ | Kuyruk riski |

### 2.4 Sampler teşhisleri (atlas)

| Teşhis | İdeal / not |
|--------|-------------|
| ESS = 1/Σp² | 84 = tam uniform; 1 = tek hücre |
| max_cell_probability | sıcak hücre baskınlığı |
| TV(uniform) | 0 = uniform |
| top10_overlap (ardışık stage) | >0.3 ön-kayıtlı “kalıcı frontier” eşiği |
| eligible_fraction | LP’sine güvenilen hücre oranı |
| \|LP\| vs SEM | SNR; <1 → gürültü kovalama |

---

## 3. Deneme haritası

Zaman çizelgesi ve “kaçıncı deneme” eşlemesi:

| # | Tarih | Atlas / dashboard id | Model run | Sampler modu | Pair? | Sonuç özeti |
|---|-------|----------------------|-----------|--------------|-------|-------------|
| 0 | 23–24 Tem | — | çoklu erken kesik (seed 7/17/19/23/…) | — | hayır | Smoke / failed starts; veri üretmedi |
| **1** | **24 Tem** | `01_Jul24_*` · job `506791` | `Jul24_13-55-22` LP + `13-55-38` UNI | fixed **β=5**, ε=0 | LP+UNI | Fiilen uniform; val “avantajı” yanıltıcı; **holdout yok** |
| **2** | **27 Tem sabah** | `02_507739` + resume `02_507758` | `Jul27_11-01-35` LP + `11-01-19` UNI | **adaptive_ess** β≈0.9–3.5 | LP+UNI | Fiilen uniform; holdout **berabere** (0.1295 vs 0.1306) |
| **3** | **27 Tem 15:07** | `03_Jul27_lpacrl_beta1_crash` | `Jul27_15-08-12` | fixed **β=1**, cap/gate **ölü** | yalnız LP | **Çöküş** ESS~2.8, maxp~0.90, 84→~54 hücre |
| **4** | **27 Tem 16:00** | `04_Jul27_lpacrl_beta1_fixed` | `Jul27_16-00-38` | fixed **β=1**, cap+gate **canlı** | yalnız LP | Sampler sağlıklı; kümülatif diyet yine uniform; holdout **0.1608 (en kötü)** |

Dashboard’lar (lokal, bu oturumda):

| Deneme | URL (varsa) |
|--------|-------------|
| V5 #1 | http://127.0.0.1:8766 |
| V5 #2 | http://127.0.0.1:8767 |
| (V6 frontier, referans) | http://127.0.0.1:8765 |

---

## 4. Atlas’tan yeniden hesaplanan sampler özeti

`logs/lpacr_dashboard_data/*/frames.ndjson` üzerinden (2026-07-29):

| Run | Frames | Step aralığı | ESS med (min–max) | maxp med (max) | TV med | top10_ovl med (max) | macro-perf first→last |
|-----|--------|--------------|-------------------|----------------|--------|---------------------|------------------------|
| #1 LP | 36 | 2k→72k | **82.3** (71.8–84.0) | 0.017 (0.029) | 0.053 | 0.00 (0.40) | 0.11 → 10.02 |
| #1 UNI | 35 | 2k→70k | **84.0** sabit | 0.012 | ~0 | 1.00 | 0.03 → 8.98 |
| #2 LP (507739) | 24 | 6k→52k | **72.6** (57.1–81.4) | 0.033 (0.057) | 0.129 | 0.00 (0.30) | 5.80 → 10.56 |
| #2 UNI (507739) | 23 | 6k→50k | **84.0** | 0.012 | ~0 | 1.00 | 5.47 → 10.33 |
| #2 LP resume (507758) | 15 | 54k→82k | **74.8** (66.7–81.6) | 0.030 (0.061) | 0.109 | 0.00 (0.10) | 9.27 → 10.40 |
| #2 UNI resume | 17 | 50k→82k | **84.0** | 0.012 | ~0 | 1.00 | 10.33 → 10.28 |
| #3 crash | 13 | 4k→42k | **2.81** (1.24–11.4) | **0.556** (**0.895**) | **0.79** | 0.10 (0.40) | 3.17 → 9.22 |
| #4 fixed | 41 | 2k→82k | **59.4** (11.1–84.0) | 0.047 (0.174) | 0.213 | **0.00 (0.30)** | 0.09 → 9.16 |

**Okuma:**

- #1 ve #2: ESS ~73–82 → sampler **uniform’a çok yakın**; top-10 overlap medyan 0 → kalıcı frontier yok.
- #3: ESS ~3, maxp 0.90 → **kaçak konsantrasyon / çöküş**.
- #4: ESS ~59, maxp ≤0.17 → **sağlıklı ama ılımlı** sapma; top-10 overlap hâlâ ≤0.30 (ön-kayıtlı “>0.3 kalıcı frontier” kriteri **tutmadı**).

---

## 5. Koşu bazında hikâye

### 5.1 Koşu #1 — fixed β = 5 (24 Tem, “ilk deneme”)

**Konfig:** `β=5`, `ε=0`, cap/gate/adaptive yok.

**Sampler:** ESS medyan 82.3 → softmax fiilen düz. `|LP|/β ≈ 0.1` mertebesi → yapısal uniform.

**Validation (yanıltıcı “zafer”):**

| iter | LP spnte | UNI spnte |
|------|----------|-----------|
| 1000 | 0.222 | 0.469 |
| 2600 | **0.127** | 0.168 |
| 3000 | 0.140 | 0.173 |

Görünür LP avantajı ~0.04 spnte. **Holdout yapılmadı.** Sonradan anlaşıldı ki:

1. İki kol da neredeyse aynı (uniform) diyet yiyordu → mekanizma farkı yok.
2. Uniform kolu bu seed’de outlier (erken spnte/success çok kötü).
3. Tek-seed varyans, iddia edilen etkiyle aynı mertebede.

**Ders:** ≤0.03 spnte farkına tek seed + yalnız validation ile inanma.

---

### 5.2 Koşu #2 — adaptive_ess (27 Tem sabah, “ikinci deneme”)

**Konfig (özet):** `β=2.5` taban, `temperature_mode=adaptive_ess`, `β∈[0.75,8]`, `ε=0.03`, `max_cell_p=0.08`, `min_stage_episodes_for_lp=16`.  
İki Slurm job’u dikişli: `507739` (step ~6k–52k) + resume `507758` (~50k–82k).

**Sampler:** ESS medyan ~73–75; `signal_quality` medyan ~0.40 → formül **bilinçli olarak uniform’a yakın** tutuyor.

**Holdout final (asıl karar metrikleri):**

| Kol | Seçilen iter | **spnte ↓** | succ | cellsucc | fall | CVaR |
|-----|--------------|-------------|------|----------|------|------|
| LP-ACRL | 3500 | **0.1295** | 0.889 | 0.762 | 0.080 | 0.464 |
| uniform | 2800 | **0.1306** | 0.892 | 0.726 | 0.082 | 0.466 |

**Fark 0.0011 → berabere.** Cuma “avantajı” tek-seed varyans olarak kapandı.

**SNR:** medyan reliability ~0.45; top-10 overlap ≈ 0; LP gürültü kovalıyor.

---

### 5.3 Ara karar — offline β replay

Uniform-rejim LP log’u ile replay:

| β | ε | Davranış |
|---|---|----------|
| 5.0 | 0.02 | ESS ~82 — ölü |
| **1.0** | 0.02 | ESS ~40–48 — “makale rejimi” gibi |
| 0.5 | 0.02 | ESS ~10 — çöküyor |

**Karar:** fixed `β=1`, `ε=0.02`, cap 0.25, adaptive kapalı.  
**Ön-kayıtlı başarı (atlas):** ESS 25–60 · top10_overlap >0.3 · koherent frontier heatmap.

---

### 5.4 Koşu #3 — fixed β = 1, bug’lı (çöküş)

**Konfig kağıt üzerinde doğruydu; kodda iki koruma fixed modda çalışmıyordu.**

| Stage / step | ESS | maxp | Gözlenen hücre (kaba) |
|--------------|-----|------|------------------------|
| ~4k | 11.4 | 0.16 | 84 |
| ~8k | **2.3** | **0.65** | ~82 |
| ~24k | **1.2** | **0.90** | ~62 |
| ~32k | 1.3 | 0.88 | ~54 |

Örnek: 1–2 episode’luk hücrede `|LP|` 6–9 → `e^9` ~9000 → mass ~%90 → flood → regresyon → **sonraki gürültü hücresine atlama**.

**Üretim bug’ları (düzeltildi):**

1. **Per-cell cap** yalnız `adaptive_ess` dalında uygulanıyordu → fixed’te ölü konfig (`maxp` 0.895).
2. **Episode gate** (`min_stage_episodes_for_lp`) yalnız adaptive skor yolundaydı → fixed’te 1-episode LP tam ağırlık; `eligible_fraction` atlas’ta sürekli **0.00** (anlamsız).
3. Dashboard: NaN → HTTP 400 → sessiz frame kaybı (`plugger` sanitizer).
4. Fixed modda `target_ess=84` sentinel’inin “hedef uniform” gibi okunması (kozmetik).

**Offline replay neden yakalamadı:** girdi LP dağılımı uniform rejimden; softmax kendi politikası altında durağan değil → kalibrasyon **kendi ürettiği rejime ekstrapole edilemez**.

---

### 5.5 Koşu #4 — fixed β = 1, düzeltilmiş (tam 3500 iter)

**Aynı hiperparametreler, düzeltilmiş kod.** 41 atlas stage, step 2k→82k, temiz bitti.

**Sampler sağlığı (başarılı):**

- ESS min 11.1, medyan ~59; maxp max 0.174 (cap 0.25 hiç bağlanmadı)
- Kapsama genelde 84/84; eligible medyan ~0.90
- Erken stage’de daralıp gate ile geri açılma (**salınım bandı**, monoton çöküş değil)

**Ama kümülatif maruziyet fiilen uniform:**

| Ölçü | Değer | Uniform |
|------|-------|---------|
| Stage-ortalama p, KL(p‖u) | ~0.012 nat | 0 |
| Gerçekleşen episode payı KL | ~0.007 nat | 0 |
| En çok / en az örneklenen hücre | ~1.65× | 1× |

Stage-içi p_max/p_min 10–100× olabiliyor; **her stage başka hücrelere** dağıtılıyor → ortalama diyet uniform.

**Neden — LP gürültünün altında:**

```
corr(LP_t, LP_{t+1})  medyan ≈ −0.45
saf gürültü teorik     = −0.50   (fark tahmini mean-reversion)
|LP| medyan ≈ 0.39  <  gürültü tabanı √2·SEM ≈ 0.51
```

Sinyal/gürültü genlik oranı ~0.5. top10_overlap 41 stage’de medyan 0, max 0.30 → **kalıcı frontier kriteri tutmadı.**

**Geçici vx curriculum:** stage 2–16’da yavaş→hızlı süpürme (r≈+0.76); stage 17+ ölüyor (politika yeterliliği artınca hücreler arası fark SEM altına iniyor). Terrain ailesinde **yön yok**, salınım var.

**Holdout (job 508168, üç koşu aynı bank/protokol):**

| Koşu | Seçilen iter | **spnte ↓** | succ | cellsucc | fall | CVaR |
|------|--------------|-------------|------|----------|------|------|
| **#4 LP fixed β=1** | 2600 | **0.1608** | 0.866 | 0.655 | 0.082 | **0.439** |
| #2 LP adaptive | 3500 | 0.1295 | 0.889 | 0.762 | 0.080 | 0.464 |
| #2 uniform | 2800 | 0.1306 | 0.892 | 0.726 | 0.082 | 0.466 |

**Curriculum’un gerçekten “çalıştığı” ilk koşu, holdout’ta en kötü koşu.**  
Dört bağımsız okuma aynı yön (LP #4 aleyhine): TB mean_reward, atlas ortak-hücre perf, validation spnte, holdout spnte. Tek olumlu nüans: CVaR biraz daha iyi (kuyruk) — tek nokta, iddia kurulmaz.

Validation #4’te 2600’den sonra bozulma (0.158 → 0.179); gürültü-kovalayan örneklemenin geç eğitim maliyetiyle uyumlu okuma.

---

## 6. Karşılaştırmalı skor kartı

### 6.1 Holdout (var olan tek sağlam final tablo)

| Sıra (spnte) | Koşu | spnte | vs uniform |
|--------------|------|-------|------------|
| 1 (en iyi) | #2 LP adaptive | 0.1295 | −0.001 (berabere) |
| 2 | #2 uniform | 0.1306 | — |
| 3 (en kötü) | #4 LP fixed β=1 | 0.1608 | **+0.030** (kötü) |
| — | #1 | — | holdout yok |
| — | #3 crash | — | eval yapılmadı / anlamsız |

### 6.2 “LP gerçekten örnekledi mi?”

| Koşu | Sampler karakteri | Eğitim diyeti | Eval |
|------|-------------------|---------------|------|
| #1 β=5 | Yapısal uniform | ≈ uniform | Val yanıltıcı |
| #2 adaptive | Bilinçli yumuşak uniform | ≈ uniform | Holdout berabere |
| #3 β=1 bug | Kaçak tek-hücre | Bozuk / daralan | — |
| #4 β=1 fix | Sağlıklı ılımlı sapma | **Kümülatif yine ≈ uniform** | Holdout en kötü |

### 6.3 Ön-kayıtlı atlas kriterleri (#4)

| Kriter | Hedef | #4 sonucu |
|--------|-------|-----------|
| ESS bandı | 25–60 | Medyan ~59 (erken 11–20, sonra 35–70) — **kısmen** |
| top10_overlap | >0.3 | Medyan 0.0, max 0.3 — **tutmadı** |
| Koherent frontier heatmap | evet | Kalıcı hücre frontier yok — **tutmadı** |

---

## 7. Bulunan bug’lar ve metodoloji dersleri

### 7.1 Kod

| Bug | Etki | Durum |
|-----|------|--------|
| Cap yalnız adaptive dalda | β=1 fixed çöküş | Düzeltildi |
| Episode gate yalnız adaptive | 1-ep LP flood | Düzeltildi |
| Dashboard NaN → 400 → frame drop | Atlas delikleri | Düzeltildi |
| Sentinel target_ess=84 fixed’te | Yanlış okuma | Raporlama bastırıldı |

### 7.2 Metodoloji

1. **Tek seed + validation-only “zafer” güvenilmez** (#1).
2. **Offline β replay, politikanın kendi rejimine ekstrapole edilemez** (#3).
3. **Gate = çöküşten kurtaran = uniform’a çeken aynı mekanizma** (nötr LP=0 imputation).
4. **84 hücre × ~50 ep/stage** bütçesinde `|LP| < gürültü tabanı` → hücre-seviyesi LP ölçülemez.
5. β’nın üç ayarı da aynı yere çıktı: (5 → uniform) · (1 gatesiz → çöküş) · (1 gate’li → ılımlı + kümülatif uniform + kötü holdout).

---

## 8. Ne biliyoruz / ne bilmiyoruz

### Kesin

- Bu setup’ta **LP-ACRL uniform üstünlüğü gösterilmedi.**
- #1–#2’de sampler pratikte çalışmadı (uniform); #3 çöktü; #4 sağlıklı sampler + **kötü/ nötr-altı final skor**.
- Kalıcı hücre frontier yok (overlap kriteri).
- Cap+gate artık gerçek ve stres testiyle sınırlı (ESS tabanı ~4.16 cap probe).

### Bilinmeyen / açık

- Çok-seed varyans bandı (hiçbir pair multi-seed değil).
- Stage length 4000+ veya havuzlanmış LP ile SNR artar mı (hipotez, test edilmedi).
- CVaR’ın #4’te biraz iyi olması sistematik mi (tek nokta).

### Tasarım sonucu (HISTORY ile uyumlu karar)

Hücre-seviyesi LP’yi bu bütçede **bırak**. Yön:

- **Faktörize 2-seviye örnekleme:** vx-bandı × terrain (vx zaten tek tutarlı erken sinyaldi)
- **Havuzlama / kayan pencere regresyon LP** (~21× episode/tahmin → SNR ~0.5 → ~2 mertebesi)
- β ince ayarını ana kaldıraç sanmayı bırak

Bu çizgi, proje içinde **V6 frontier** (success-gated speed×level, LP yok) denemesine de zemin hazırladı.

---

## 9. Artifact dizini (hızlı referans)

### Atlas / dashboard

```
logs/lpacr_dashboard_data/
  01_Jul24_go2_v5_lpacrl-seed1/          # koşu #1 LP
  01_Jul24_go2_v5_uniform-seed1/         # koşu #1 UNI
  02_507739_20260727_100500_*            # koşu #2 ilk segment
  02_507758_20260727_110022_*            # koşu #2 resume
  03_Jul27_lpacrl_beta1_crash/           # koşu #3
  04_Jul27_lpacrl_beta1_fixed/           # koşu #4
logs/curriculum_atlas{,_local}/          # ham job kopyaları
```

### Eğitim / eval

```
logs/go2_v5_ued/Jul24_13-55-{22,38}_*   # #1
logs/go2_v5_ued_tb/Jul27_11-01-{19,35}_* # #2 (+ heldout_final)
logs/go2_v5_ued_tb/Jul27_15-08-12_*      # #3
logs/go2_v5_ued_tb/Jul27_16-00-38_*      # #4
```

### Ayrıntılı kronoloji

```
lpacr/HISTORY.md          # gün gün, formül, bug patch, stage tabloları
lpacr/V5_DENEME_OZETI.md  # bu dosya
```

---

## 10. Tek bakışlık tablo (yönetici özeti)

| Deneme | Sampler | Holdout spnte | Yorum |
|--------|---------|---------------|--------|
| #1 β=5 | ≈ uniform | yok | Val “zafer” yanıltıcı |
| #2 adaptive | ≈ uniform | 0.130 (LP) / 0.131 (UNI) | **Berabere** — en temiz LP vs UNI karşılaştırması |
| #3 β=1 bug | çöküş | — | Cap+gate fixed’te ölüydü |
| #4 β=1 fix | sağlıklı, ılımlı | **0.161** | Gerçek curriculum diyet olarak yine ≈U; skor en kötü |

**Son cümle:** V5 kampanyası LP-ACRL’i “çalıştırmayı” öğretti (bug’lar, β, gate, ESS teşhisleri) ama **gösterilemedi** ki hücre-seviyesi LP bu task space ve bütçede uniform’dan daha iyi politika üretir. Sonraki iş, LP’yi daha az gürültülü bir temsile taşımak (havuzlama / faktörizasyon) veya V6’daki gibi success-gated frontier gibi farklı bir curriculum primitifine geçmek.
