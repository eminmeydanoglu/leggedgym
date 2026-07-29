# Alt-yüklenici promptu — V5 LP-ACRL veri madenciliği + HTML rapor

> Bu dosya bir **prompt**tur. Aşağıdaki metnin tamamını analiz ajanına ver.

---

## 0. Rolün ve tek cümlelik görevin

Sen deneysel RL/istatistik analistisin. Görevin: **hiç yeni eğitim koşmadan**, V5 LP-ACRL
kampanyasının kayıtlı verisinden çıkarılabilecek her şeyi çıkarmak ve tek bir
**self-contained HTML raporda** sunmak.

Cevaplanması gereken üç soru, önem sırasıyla:

1. **Bir LP (learning-progress) sinyalinin "iyi" olup olmadığı nasıl ölçülür?**
   Yeniden kullanılabilir, null'ları türetilmiş bir *LP kalite skor kartı* tanımla ve
   eldeki bütün koşulara uygula.
2. **Bizim LP'mizin "gürültülü" olduğu nasıl ölçülür — ve gürültü hangi bileşenlerden
   oluşuyor?** Sayısal bir **gürültü bütçesi** çıkar (her bileşenin toplam varyanstaki payı).
3. **Gürültüyü hangi müdahale çözer?** Aday müdahaleleri (daha çok episode, daha uzun
   stage, vy/yaw komutlarını sabitleme, kovaryat düzeltmesi, havuzlama/faktörizasyon,
   metrik değişimi, tahminci değişimi) **mevcut veriyle ölçerek** sırala. Her biri için
   "beklenen kazanç" ver, tahmin değil; ölçülemiyorsa neden ölçülemediğini yaz.

Ek olarak: bir sonraki kampanyanın **güç analizini** yap (kaç seed, kaç iterasyon, hangi
etki büyüklüğü tespit edilebilir).

**Uygulama yapmıyorsun.** Eğitim kodunu değiştirme, yeni koşu başlatma, GPU kullanma.
Yazacağın tek kod: `lpacr/analysis/` altına giren offline analiz scriptleri + rapor üretici.

---

## 1. Ortam

- Çalışma dizini: `/home/emin/code/online-estimation/genesis-wp/LeggedGym-Ex`
- Python: **`.venv/bin/python`** (repo kökündeki venv). numpy 2.4, scipy 1.17, pandas 3.0,
  tensorboard mevcut. Sistem `python3` ve `~/.local/bin/pytest` kullanma.
- GPU yok, simülatör yok, ağ yok. Her şey kayıtlı dosyalardan.
- Mevcut analiz paketi: `lpacr/analysis/` — `atlas.py` ortak yükleyici, kullan ve genişlet.

---

## 2. Veri envanteri (yollar doğrulandı, 2026-07-29)

### 2.1 Curriculum Atlas frame akışları — `logs/lpacr_dashboard_data/<run>/frames.ndjson`

Stage başına bir satır JSON. `metrics.*` alanları 84 elemanlı liste
(`task_id = vx_bin * 21 + terrain_cell`).

| dizin | koşu | algo | frame | step aralığı | not |
|---|---|---|---|---|---|
| `01_Jul24_go2_v5_lpacrl-seed1` | #1 LP (β=5) | lp_acrl | 36 | 2k–72k | **`performance_sem`, `eligible_for_lp`, `previous_stage_episode_count` YOK** |
| `01_Jul24_go2_v5_uniform-seed1` | #1 UNI | uniform | 35 | 2k–70k | aynı eksik alanlar |
| `02_507739_..._lpacrl-seed1` | #2 LP (adaptive) | lp_acrl | 24 | 6k–52k | tam şema |
| `02_507739_..._uniform-seed1` | #2 UNI | uniform | 23 | 6k–50k | tam şema |
| `02_507758_..._lpacrl-seed1` | #2 LP resume | lp_acrl | 15 | 54k–82k | tam şema |
| `02_507758_..._uniform-seed1` | #2 UNI resume | uniform | 17 | 50k–82k | tam şema |
| `03_Jul27_lpacrl_beta1_crash` | #3 çöküş | lp_acrl | 13 | 4k–42k | + `top10_overlap_prev`, `tv_distance_uniform`, `lp_reliability_median` |
| `04_Jul27_lpacrl_beta1_fixed` | #4 fixed β=1 | lp_acrl | 41 | 2k–82k | tam şema, §11'in tek kaynağı |
| `v6-frontier-1500-seed1-20260728` | V6 referans | frontier | 8 | — | **farklı şema** (success-gated), opsiyonel karşılaştırma |

`metrics` anahtarları (tam şema): `performance`, `performance_sem`, `learning_progress`,
`effective_learning_progress`, `eligible_for_lp`, `previous_stage_episode_count`,
`sampling_probability`, `stage_episode_count`, `task_assignment_count`,
`task_completion_count`.

`metadata.frame.diagnostics`: `entropy`, `effective_sample_size`, `max/min_cell_probability`,
`temperature_mode`, `effective_beta`, `target_ess`, `signal_quality`, `eligible_cell_count`,
`eligible_fraction`, `ess_guard_uniform_mix`, `task_assignment_coverage`,
`completed_outcome_coverage`, `late_outcome_count`, `top10_overlap_prev`,
`tv_distance_uniform`, `lp_reliability_median`, `sampled_lp_mass`.
`metadata.frame.standstill`: rezerve standstill episode sayısı/return/uzunluk.

Aynı atlaslar `logs/curriculum_atlas{,_local}/<job>/` altında ham kopya olarak da var
(`505449_20260727_155936` = koşu #4, `atlas.py`'nin DEFAULT_ATLAS'ı).

> ⚠️ `atlas.load()` şu an sadece tam şemayı okuyor; koşu #1'i açmak için yükleyiciyi
> eksik-alan toleranslı hale getir (eksik alanı `NaN` doldur, hangi analizin o koşuda
> yapılamadığını raporda açıkça yaz).

> ⚠️ `BOOTSTRAP_FRAME = 0` normal stage değil (episode sayısı ~5×, LP tanımsız). Atla.

### 2.2 **Per-cell / per-replica validation bankası** — kampanyanın en değerli ve **§11'de hiç kullanılmamış** verisi

```
logs/go2_v5_ued/Jul24_13-55-22_v5_lp_acrl_genesis_seed1/ued_validation/model_{1000,1400,1800,2200,2600,3000}.json
logs/go2_v5_ued/Jul24_13-55-38_v5_uniform_genesis_seed1/ued_validation/model_{1000,...,3000}.json
```

Her dosya: `measurements` = **1008 satır = 84 hücre × 12 replika**, alanlar
`cell_id, replica_id, spnte_lin, spnte_yaw, tracking_ang_err, fall_rate, survival_steps,
command_vx, command_vy, command_yaw, geometry_hash`. Ayrıca `scores.cells` (84 hücre
özeti: `spnte_lin, fall_rate, replica_success_rate, cell_success`) ve makro skorlar.

**Doğrulanmış kritik özellikler:**

- **Tam eşleşmiş tasarım:** `(cell_id, replica_id) → (command_vx, command_vy, command_yaw)`
  üçlüsü **bütün checkpoint'lerde ve her iki kolda birebir aynı**
  (`validation_bank_fingerprint` ortak). Yani checkpoint'ler arası fark ve kollar arası
  fark **matched-pairs**; komut gürültüsü farkta sadeleşiyor.
- `command_vy, command_yaw ∈ [-1, 1]` — hücreyi tanımlayan `vx` bandının genişliği ise
  yalnız 0.3–0.5 m/s. Yani **hücre-içi komut heterojenliğinin baskın kısmı vy/yaw**.
- Hücre-içi `spnte_lin` CV medyanı ≈ 0.45 (12 replika ⇒ hücre SEM/ortalama ≈ 0.13).
- Makro spnte eğrileri (iter 1000→3000):
  LP kolu `0.2219, 0.1790, 0.1519, 0.1493, 0.1265, 0.1397`;
  UNI kolu `0.4686, 0.3060, 0.2003, 0.1866, 0.1680, 0.1729`.

> Bu banka **yalnız koşu #1'in iki kolu için** yerelde mevcut. Koşu #2/#4'ün
> `heldout_final/` ve `ued_validation/` dizinleri yerelde **boş** — holdout sayıları
> sadece `V5_DENEME_OZETI.md` ve `HISTORY.md` içindeki tablolarda yazılı. Bunu bir veri
> boşluğu olarak raporla; eksik dosyayı uydurma.

### 2.3 TensorBoard eğitim eğrileri

`logs/go2_v5_ued_tb/<run>/events.out.tfevents.*` ve `logs/go2_v5_ued/<run>/...`
25 skaler: `Train/mean_reward`, `Train/mean_episode_length`, 14× `Episode/rew_*` bileşeni,
`Loss/*`, `Policy/mean_noise_std`, `Perf/*`. İterasyon başına bir nokta.

| koşu | dizin | iter |
|---|---|---|
| #1 LP | `logs/go2_v5_ued/Jul24_13-55-22_v5_lp_acrl_genesis_seed1` | 3000 |
| #1 UNI | `logs/go2_v5_ued/Jul24_13-55-38_v5_uniform_genesis_seed1` | 3000 |
| #2 LP | `logs/go2_v5_ued_tb/Jul27_11-01-35_v5_lp_acrl_genesis_seed1` | 1300 |
| #2 UNI | `logs/go2_v5_ued_tb/Jul27_11-01-19_v5_uniform_genesis_seed1` | 1300 |
| #3 crash | `logs/go2_v5_ued_tb/Jul27_15-08-12_v5_lp_acrl_genesis_seed1` | ~1700 |
| #4 fixed | `logs/go2_v5_ued_tb/Jul27_16-00-38_v5_lp_acrl_genesis_seed1` | 3500 |

`run_manifest.json` her koşuda var (git commit, DR dağılımı, `max_iterations`,
checkpoint seçim protokolü). **Sampler hiperparametreleri manifest'te yok**; β/ε/cap/gate
değerlerini `HISTORY.md` + atlas `curriculum_config_fingerprint` + `diagnostics.effective_beta`
üzerinden eşle.

### 2.4 Birim dönüşümleri (kullanmadan önce doğrula)

- 1 iterasyon = 24 control step (3000 iter ↔ 72k step ile doğrulandı).
- 1 stage = 2000 control step ≈ **83 iterasyon**.
- Bir stage'de hücre başına kabul edilen episode ≈ 41 (sansürsüz ≈ 83).

### 2.5 Yazılı bağlam

- `lpacr/V5_DENEME_OZETI.md` — kampanyanın anlatı özeti (koşu #1–#4, holdout tabloları).
- `lpacr/HISTORY.md` — gün gün kronoloji; **§10 ve özellikle §11 mutlaka oku**.
- `logs/go2_v5_ued/v5_pair_seed1_data_report.md` — koşu #1 için yorumsuz veri dökümü
  (validation tabloları terrain/level/vx kırılımlarıyla).
- `lpacr/analysis/README.md` — mevcut teşhis paketinin sözleşmesi.

---

## 3. Zaten bilinen — tekrarlama, doğrula ve genişlet

`HISTORY.md §11` (28 Temmuz analiz turu) **yalnız koşu #4 atlası üzerinde** yapıldı.
Ana bulguları:

| § | bulgu |
|---|---|
| 11.0 | **Kural: null'ı türetilmemiş bir sayı kapı olarak kullanılamaz.** |
| 11.1 | Stage sansürü: completion'ların **%46'sı** LP'ye alınmıyor (`assigned_revision != sampler_revision`); sansür rastgele değil, `corr(late_fraction, performance) = +0.705`. |
| 11.2 | İki sinyal kapısı: `α_SEM = clip((Var_cells(LP) − E[LP_SE²]) / Var_cells(LP), 0, 1)`, `α_temporal = clip(1 + 2·corr(LP_t, LP_{t+1}), 0, 1)`. 40 stage'in 27'sinde α_SEM = 0. Erken (2–16) α≈0.23, geç (17–41) α≈0.00. |
| 11.3 | Havuzlama α_SEM'i 0.96'ya çıkarıyor ama α_temporal'ı **hiç** iyileştirmiyor (0.108→0.103) ⇒ α_SEM **ortak-mod (policy-update) gürültüsüne kör**. Kapı `min(α_SEM, α_temporal)` olmalı. |
| 11.4 | Kayıtlı `lp_reliability = |LP|/(|LP|+lp_sem)` metriğinin **saf gürültü sabit noktası 0.444**; ölçülen 0.414 ⇒ hiçbir zaman sinyal ölçmemiş. |
| 11.5 | `z = LP/MAD(LP)` normalizasyonu tek başına mayın: β=1'de max/min softmax oranı medyan 273×, en kötü 5.5e5×. Skorlama üç katmanlı olmalı: ölçek (z+clip) · sertlik (β) · kapı (α). |
| 11.6 | Öngörü testinin null'ı pencere tasarımına bağlı. Ayrık pencerelerle: lag1 **+0.083**, lag2 +0.042, lag3 ≈0 ⇒ **sinyal var ama ömrü ~1 stage (~83 iter)**, sampler'ın tepki gecikmesinden kısa. Asıl hedef korelasyonu büyütmek değil **ufku uzatmak**. |
| 11.7 | Episode sayısı **bağlayıcı kısıt değil**: geç rejimde `σ²_signal ≈ 0`; 8× bütçe stage başına yalnız ~+3.7 hücre kazandırıyor. `sem ~ N^b`: erken b=−0.33 (varyans tabanı SE'nin %90'ı), geç b=−0.47 (temiz). Düşük-N hücreleri geç rejimde top-10 p'nin %48'ini oluşturuyor ama `corr(N, z) ≈ 0` ⇒ fazla `|LP|` tam olarak fazla gürültü. Döngü şu an **negatif geri besleme** (`p[t]→N[t+1]` ρ=+0.94, `N→|LP|` ρ=−0.14) — koşu #4'ün çökmeme sebebi bu. Gate (`min_stage_episodes_for_lp=16`) sinyal kaybetmeden gürültü eliyor (`|LP|` 2× büyük, `z` aynı, p=0.11). |
| 11.7 anomali | `performance_sem` fazla tahmin ediliyor olabilir: hücre-içi `vy`/`omega_z` komut heterojenliği iki stage'de de var, farkta sadeleşiyor, ama `LP_SE = √(sem_t² + sem_{t−1}²)` onu **iki kez sayıyor**. |

**Senden beklenen tutum:**

- Bu bulguları **diğer 5 koşuya taşı** (özellikle uniform kollara) ve tutarlı mı bak.
- Bir bulgu yanlışlanıyorsa **açıkça yaz**: "§11.x koşu #4'te X diyordu; koşu #2 UNI'da Y çıktı; sebebi Z."
- §11'in ulaşamadığı **dış ölçüt** artık elinde: §2.2 validation bankası. §11'in bütün
  self-consistency teşhisleri (LP'yi kendi geleceğiyle karşılaştırmak) bununla
  **kriter geçerliliği** testine yükseltilebilir.

---

## 4. Analiz backlog'u

Her madde şu formatta raporlanacak: **soru · yöntem · null · veri · sonuç · karar**.
Null'ı türetilmemiş hiçbir sayıyı eşik olarak kullanma (§11.0).

### A. LP kalite skor kartı (normatif çerçeve) — **birinci öncelik**

Amaç: "bu LP iyi mi?" sorusunu cevaplayan, **her gelecekteki tahminciye uygulanabilir**
bir metrik seti tanımla. Her metriğin (i) tanımı, (ii) saf-gürültü null'ı, (iii) geçme
eşiği, (iv) eldeki 6 koşudaki değeri olacak. Önerilen katmanlar:

- **A1 — Güvenilirlik (reliability).** Kesitsel varyansın ne kadarı gerçek sinyal:
  `α_SEM`, `α_temporal`, `α = min(...)`. §11.2'yi bütün koşulara uygula, rejim (erken/geç)
  ve koşu kırılımıyla. Uniform kollarda ayrıca hesapla (aşağı bak: C1).
- **A2 — Test-tekrar test (split-half) güvenilirliği.** Aynı gerçek LP alanının iki
  bağımsız ölçümü ne kadar korele? Atlas'ta per-episode veri olmadığı için stage-içi
  split-half yapılamıyor; **onun yerine C2'deki kollar-arası replikayı kullan.**
- **A3 — Zamansal kalıcılık ve sinyal ufku.** Ayrık pencerelerle ACF(lag 1..5); sinyal
  yarı ömrünü **stage ve iterasyon cinsinden** ver. Null'ı analitik olarak türet
  (paylaşılan pencere ⇒ −0.5) ve **hücre-içi blok permütasyonu** ile doğrula
  (hücre etiketi permütasyonu bu artefaktı yakalamaz — §11.6).
- **A4 — Öngörü/kriter geçerliliği (dış ölçüt).** Eğitim-zamanı LP, **bağımsız ölçülmüş**
  gelecekteki iyileşmeyi öngörüyor mu? Veri: §2.2. Yöntem: atlas LP'sini checkpoint
  pencerelerine (1000–1400, …, 2600–3000) topla; hücre başına validation `spnte_lin`
  iyileşmesini hesapla; Spearman + permütasyon null. **Hem eşzamanlı hem ileri-dönük
  (bir pencere kaydırılmış) versiyonu ver** — ikincisi asıl testtir.
  *(Ön sondaj: eşzamanlı Spearman pencere başına +0.15, +0.11, +0.13, −0.20, +0.15 —
  yani sıfırdan ayırt edilemez düzeyde. Bunu doğru null'ı, güven aralığı ve pencere
  birleştirmesiyle düzgün yap; ön sondajı sonuç olarak kopyalama.)*
- **A5 — Sıralama kararlılığı.** top-k overlap (k=5,10,20) ardışık ve uzak stage'ler
  arasında; null = hipergeometrik beklenen örtüşme (k=10, n=84 ⇒ 0.119). Kampanyanın
  ön-kayıtlı ">0.3" kriteri bu null'a göre yeniden yorumlansın.
- **A6 — Karar-teorik değer (asıl kapı).** Ölçülen α ve ufukla, **herhangi bir**
  sampler'ın kazanabileceği üst sınır nedir? Yöntem: ölçülen sinyal yapısını
  (α, ACF, hücre-içi varyans, tepki gecikmesi) parametreleştirip sentetik bir
  bandit/curriculum simülasyonu koş; oracle sampler (gerçek LP'yi bilen) ile uniform
  arasındaki farkı, ve gürültülü LP tahmincisinin bu farkın yüzde kaçını yakaladığını
  ver. **Çıktı: "bu bütçede LP kovalamanın teorik tavanı ≈ X" tek sayısı.**
- **A7 — Doygunluk / tavan testi.** Hücreler arasında *gerçek* öğrenilebilirlik farkı
  hâlâ var mı? Hücre başına validation eğrilerini (6 checkpoint) fitle, kalan headroom'un
  hücreler arası dağılımını ver. Dispersiyon ≈ 0 ise **uniform doğru cevaptır** ve bu bir
  başarısızlık değil ölçülmüş bir sonuçtur — öyle yaz.

### B. Gürültü bütçesi — **birinci öncelik**

Amaç: `Var(LP_ölçülen)` toplamını bileşenlere ayır, her birinin **yüzde payını** ver.
Aday bileşenler:

- **B1 — Hücre-içi episode örnekleme gürültüsü** (`performance_sem`'in gördüğü kısım).
- **B2 — Ortak-mod / policy-update gürültüsü** (bütün hücreleri birlikte kaydıran).
  Yöntem: `performance[stage, cell]` panelinde iki-yönlü varyans ayrıştırması
  (stage FE + cell FE + etkileşim + artık). Saf stage FE kesitsel sıralamayı bozmaz;
  **heterojen** ortak-mod bileşenini ayrıca izole et. §11.3'ün α_SEM/α_temporal
  çelişkisini bu ayrıştırmayla nicelendir.
- **B3 — Hücre-içi komut heterojenliği (vx bandı içi + vy + yaw).** İki bağımsız yol:
  1. **Doğrudan (validation bankası):** hücre-içi `spnte_lin` varyansının ne kadarı
     `command_vx, command_vy, command_yaw` ile açıklanıyor? Hücre FE + komut
     regresörleri, doğrusal **ve** doğrusal olmayan (spline / |vy| / |yaw| terimleri),
     her checkpoint için ayrı. Sonra: bu bileşen kaldırılsa hücre SEM'i ne kadar düşerdi,
     ve **α ne kadar yükselirdi?**
     *(Ön sondaj, model_3000, doğrusal: R² = 0.041 toplam — vx 0.020, vy 0.014, yaw 0.008.
     Yani "vy/yaw'ı sabitlersek gürültü çözülür" hipotezi **ilk bakışta zayıf**. Bunu
     çürüt ya da doğrula: doğrusal olmayan terimler, erken checkpoint'ler, `fall_rate`
     ve `survival_steps` üzerinden, ve training-return proxy'siyle.)*
  2. **Dolaylı doğal deney (atlas):** `vx_bin 0` bandı 0.3 m/s genişliğinde, diğer üçü
     0.5 m/s. Komut heterojenliği SEM'i şişiriyorsa, eşleşmiş N'de `vx_bin 0`
     hücrelerinin `performance_sem`'i **sistematik olarak düşük** olmalı. Test et
     (N kontrollü, terrain FE'li regresyon). Aynısını terrain zorluğu için de yap.
- **B4 — Sansür/kompozisyon gürültüsü.** §11.1'i bütün koşulara taşı; `late_outcome_count`
  ve `completed_outcome_coverage` üzerinden stage başına sansür oranının **zaman içindeki
  oynaklığının** LP'ye kattığı varyansı tahmin et (sansür oranı stage'ler arası
  değişiyorsa fark metriğine doğrudan gürültü enjekte eder).
- **B5 — Geri besleme kaynaklı gürültü (yalnız LP kollarında).** Örnekleme olasılığı
  N'i, N gürültüyü, gürültü LP'yi belirliyor. Uniform kollarda bu bileşen **yok**;
  LP ve UNI kollarının gürültü bütçelerini karşılaştırarak payını ölç.

**Çıktı: tek bir yığılmış varyans grafiği + tablo**, erken/geç rejim ve koşu kırılımıyla.
Bileşenler tam ayrıştırılamıyorsa üst/alt sınır ver, uydurma.

### C. Kullanılmamış kontroller — **en yüksek bilgi getirisi**

- **C1 — Uniform kol = temiz gözlemsel null.** Uniform kollarda da `learning_progress`
  kaydediliyor ama **kullanılmıyor**: geri besleme döngüsü yok, N hücreler arasında
  neredeyse eşit. Yani LP'nin *ölçüm* kalitesi burada confounding olmadan ölçülebilir.
  A1–A5'in tamamını uniform kollarda tekrarla ve LP kollarıyla karşılaştır.
  **Kritik soru: LP'nin ölçülemez olması sampler'ın kendi yarattığı rejimin mi
  (heterojen N, düşük-N amplifikasyonu), yoksa görevin doğasının mı sonucu?**
- **C2 — Kollar-arası replika (gerçek test-tekrar test).** Koşu #1 ve #2'de **her iki kol
  da fiilen uniform diyet yedi** (ESS 73–84). Aynı seed, aynı task space, aynı adımlar.
  Dolayısıyla LP kolu ile UNI kolunun aynı step'teki LP alanları, **aynı gerçek sinyalin
  iki bağımsız ölçümü** sayılabilir. `corr(LP_cell^{LP-arm}(t), LP_cell^{UNI-arm}(t))`
  → doğrudan güvenilirlik tahmini, hiçbir model varsayımı olmadan.
  Aynısını `performance` alanı için de yap (bu daha yüksek çıkmalı — kontrast anlamlı).
  Null: aynı koşu içinde kaydırılmış stage eşleştirmesi.
  ⚠️ İki kolun politikaları zamanla ayrışıyor; bunu bir üst sınır değil **alt sınır**
  olarak yorumla ve step ilerledikçe korelasyonun nasıl azaldığını göster.
- **C3 — Kollar-arası eşleşmiş validation replikası.** §2.2'de her iki kolun 1008
  ölçümü **aynı komut ve geometriyle** yapılmış. Kol farkını replika bazında eşleştirerek
  (paired) hesapla: eşleşmemiş teste kıyasla varyans ne kadar düşüyor, ve iki kol
  arasındaki fark hangi checkpoint'te istatistiksel olarak anlamlı? Bu, kampanyanın
  "tek seed varyansı" iddiasının doğrudan testidir.
- **C4 — Run-to-run gürültü tabanı ve MDE.** Koşu #1 ve #2'nin iki kolu da fiilen uniform
  örnekledi ⇒ aralarındaki fark **mekanizma farkı değil, saf koşu-arası gürültüdür**.
  Bu farkları (validation spnte eğrileri, TB `Train/mean_reward`, atlas ortak-hücre perf)
  kullanarak koşu-arası standart sapmayı tahmin et, sonra:
  **Δspnte = 0.005 / 0.01 / 0.02 / 0.03 etkisini %80 güçle görmek için kaç seed gerekir?**
  Kampanyanın gözlediği "avantajlar" (#1'de 0.04, #2'de 0.001, #4'te −0.030) bu tabana
  göre nerede duruyor? **Bu tablo bir sonraki kampanyayı doğrudan tasarlar.**

### D. Müdahale sıralaması — "gürültüyü ne çözer?"

Her müdahale için: **mevcut veriyle ölçülen** beklenen kazanç (α'da, ufukta, veya
A6'nın karar-teorik değerinde), maliyet, ve karar.

- **D1 — Daha çok episode / stage başına N.** §11.7'yi doğrula ve **bütün koşulara** taşı.
- **D2 — Daha uzun stage (asıl yeni test).** Atlas'ta **ardışık stage'leri birleştirerek**
  2×, 3×, 4× uzunlukta sentetik stage'ler üret (performance'ı N-ağırlıklı ortala, SEM'i
  buna göre birleştir) ve A1/A3/A4'ü her agregasyon seviyesinde yeniden hesapla.
  Bu, ekstrapolasyon değil **doğrudan ölçüm**: 2000 → 4000 → 8000 control step stage
  uzunluğunda α ve sinyal ufku ne oluyor? Tepki gecikmesi de aynı oranda büyüdüğü için
  **net kazanç** (α × ACF(gecikme)) grafiğini ver — optimum stage uzunluğu bir iç nokta
  olabilir.
- **D3 — vy/yaw komutlarını sabitleme / daraltma.** B3'ün sonucundan türet. Sabitlemenin
  **maliyetini** de yaz (politika artık o komut dağılımında eğitilmiyor; validation bankası
  vy/yaw'ı örneklemeye devam ediyor ⇒ dağılım kayması).
- **D4 — Kovaryat düzeltmesi (residualization).** Episode gerektirmeyen varyans azaltma:
  her episode'un getirisini kendi komutuna göre residualize et. B3.1 bunun **tavanını**
  verir. Mevcut atlas per-episode veri içermediği için bunu validation bankası üzerinden
  simüle et ve "aynı veriyle LP_SE ne kadar düşer" sayısını ver.
- **D5 — Havuzlama / faktörizasyon (vx × terrain).** §11.3'ün uyarısını bütün koşularda
  test et. Ek olarak **açık faktörize model** kur: `LP[c] = a[vx(c)] + b[terrain(c)] + ε`;
  marjinal bileşenlerin güvenilirliğini (A1–A3) hücre-seviyesiyle karşılaştır; katkı
  modelinin açıkladığı varyans payını ver (etkileşim gerçekten gerekli mi?).
- **D6 — Metrik değişimi.** Ham stage return doygunlaşmış olabilir. Atlas'ta v5 için
  per-hücre episode uzunluğu **yok** (v6'da var) ⇒ reward-per-step yeniden kurulamaz;
  bunu veri boşluğu olarak yaz. Bunun yerine validation bankasındaki alternatif
  metriklerin (`spnte_lin`, `fall_rate`, `survival_steps`, `replica_success_rate`)
  hücreler arası **ayırt ediciliğini ve zamansal kalıcılığını** karşılaştır: hangi metrik
  daha uzun ufuklu bir LP verir? Bu, bir sonraki tahminci için metrik seçimini belirler.
- **D7 — Tahminci değişimi.** Kod `lp_estimator ∈ {"stage", "rolling_completion"}`
  destekliyor (`legged_gym/utils/ued/episode_curriculum.py`). Atlas üzerinden kayan
  pencere / regresyon eğimi / EWMA tabanlı LP tahmincilerini **offline** üret ve
  A1–A4 skor kartında yarıştır. Kazanma kriteri korelasyon değil **ufuk** (§11.6).

### E. Sampler dinamiği ve kararlılık

- **E1 — Koşu #3'ün çöküşünü bir tasarım kuralına çevir.** `p[t] → N[t+1] → |LP|[t+1] → p[t+1]`
  döngüsünün **kazancını** ölç (her adımın elastisitesi, log-log eğim). Koşu #4'te kazanç
  <1 (negatif geri besleme, §11.7), koşu #3'te >1. **Kararlılık sınırını β, cap, gate ve
  N'nin fonksiyonu olarak yaz** — offline β replay'in yapamadığı şey buydu (§11'de
  "kendi rejimine ekstrapole edilemez" denmişti; ölçülen loop gain bu sorunu çözer).
- **E2 — Tepki gecikmesi bütçesi.** LP stage t'de ölçülüyor, p stage t+1'de uygulanıyor,
  N stage t+1'de gerçekleşiyor, etkisi t+2'de görülüyor. Ölçülen ACF ile birleştirip
  **"eyleme dönüşebilen sinyal" = α × ACF(gecikme)** oranını ver. Gecikmeyi 1 stage
  azaltmanın (ör. stage-içi güncelleme) kazancını sayısallaştır.
- **E3 — Diyet muhasebesi.** Bütün koşular için kümülatif episode payı, KL(p‖u),
  stage-ortalama p'nin KL'i, en çok/en az örneklenen hücre oranı; ESS/TV/entropi
  zaman serileri. #4'ün "stage-içi keskin ama kümülatif uniform" olgusunu
  **stage-içi vs stage-arası varyans ayrıştırmasıyla** nicelendir.

### F. Sonuç tarafı (eğitim çıktısı)

- **F1 — Eğitim eğrilerinin adil karşılaştırması.** TB `Train/mean_reward` LP vs UNI,
  ama **diyet farklı olduğu için ham karşılaştırma yanlıdır** (LP kolu zor hücreleri daha
  çok örneklerse mean_reward düşer). Bu yanlılığı atlas'taki gerçekleşen diyetle
  yeniden ağırlıklandırarak düzelt (uniform-ağırlıklı ortalama return) ve düzeltilmiş
  eğrileri karşılaştır. Düzeltmenin ne kadar değiştirdiğini göster.
- **F2 — Ödül bileşeni teşhisi.** 14 `Episode/rew_*` bileşeni: LP vs UNI kollarında hangi
  bileşenler ayrışıyor? Curriculum'un davranışa etkisi varsa burada iz bırakmalı.
- **F3 — Checkpoint seçim protokolünün etkisi.** Koşu #1'de kollar farklı iterasyonlarda
  seçildi (LP 2600 vs UNI 2800/3000). Validation eğrileri gürültülüyse "en iyi
  checkpoint" seçimi **yukarı yönlü seçim yanlılığı** yaratır. Bu yanlılığın büyüklüğünü
  eldeki 6 checkpoint'lik eğriden tahmin et (max-of-k yanlılığı) — kampanyanın raporladığı
  farkların ne kadarı bu olabilir?

### G. Opsiyonel (zaman kalırsa)

- **G1 — V6 frontier karşılaştırması.** `v6-frontier-1500-seed1-20260728` farklı bir
  curriculum primitifi (success-gated). Şeması farklı; ama **sampler kalitesi
  metriklerinden** (kalıcılık, top-k overlap, diyet konsantrasyonu) ortak olanları
  hesaplayıp LP-ACRL ile yan yana koy. "Farklı primitif daha kalıcı bir frontier
  üretiyor mu?" sorusu bir sonraki fazın gerekçesi.

---

## 5. Öncelik sırası (zaman kısıtlıysa)

1. **C1, C2, C4** — kullanılmamış kontroller; en yüksek bilgi getirisi, en düşük risk.
2. **A4, A7** — dış ölçütle kriter geçerliliği ve doygunluk (kampanyanın asıl sorusu).
3. **B (tam gürültü bütçesi)** — özellikle B3 (kullanıcının doğrudan sorduğu vy/yaw).
4. **D2, D1** — stage birleştirme + episode ölçekleme (müdahale sıralaması).
5. **A6, E1, E2** — karar-teorik tavan ve kararlılık kuralı.
6. Kalanlar.

---

## 6. Teslimat

### 6.1 Ana çıktı: `lpacr/analysis/report/v5_lp_analysis.html`

**Tek dosya, self-contained.** Harici CDN/font/script yok; CSS ve JS gömülü; grafikler
inline SVG (veya gömülü base64 PNG). Açık/koyu tema uyumlu. Yatay taşma yok
(geniş tablolar kendi `overflow-x:auto` kabında). Dil: **Türkçe** (repo dokümanlarıyla
tutarlı; teknik terimler İngilizce kalabilir).

Yapı:

1. **Yönetici özeti** — en fazla 10 madde; her biri "bulgu → sayı (null'ıyla) → karar".
   Kararlar net olsun: *yap / yapma / önce şunu ölç*.
2. **Veri envanteri ve güven notu** — hangi koşuda hangi alan var, hangi analiz nerede
   yapılamadı, veri boşlukları (holdout json'ları yerelde yok vb.).
3. **LP kalite skor kartı** — 6 koşu × metrikler matrisi; her hücrede değer, null,
   ve geçti/kaldı işareti. Raporun kalıcı ürünü bu tablo.
4. **Gürültü bütçesi** — yığılmış varyans grafiği + tablo, rejim kırılımıyla.
5. **Müdahale sıralaması** — beklenen kazanç × maliyet tablosu, sıralı.
6. **Kampanya tasarımı** — MDE/güç tablosu (seed sayısı × tespit edilebilir etki).
7. **Koşu bazında ekler** — diyet muhasebesi, sampler zaman serileri, heatmap'ler.
8. **§11 ile mutabakat** — hangi bulgu doğrulandı, hangisi yanlışlandı, hangisi genişledi.
9. **Metodoloji eki** — her null'ın türetimi (formül düzeyinde), varsayımlar, sınırlar.

Grafik beklentileri (asgari): per-hücre LP heatmap zaman serisi; α'nın stage boyunca
yörüngesi (koşu kırılımlı); ACF + null bandı; kriter-geçerlilik saçılımı; gürültü bütçesi
yığılmış bar; MDE eğrisi; diyet KL zaman serisi; stage-birleştirme kazanç eğrisi.

> Grafik/renk/tipografi kararları için `dataviz` skill'ini oku ve uygula. Her grafik
> tek başına okunabilir olsun (başlık + eksen birimi + null referans çizgisi).

### 6.2 Yan çıktılar

- **Kod:** `lpacr/analysis/` altına, mevcut stille tutarlı, yeniden çalıştırılabilir
  modüller. `atlas.py`'yi çok-koşulu ve eksik-alan toleranslı hale getir. Her scriptin
  başında ne hesapladığı ve null'ının ne olduğu yazsın. Raporu üreten script tek komutla
  çalışsın; komutu raporun sonuna yaz.
- **Ham sonuçlar:** `lpacr/analysis/report/results.json` — HTML'deki her sayı buradan
  gelsin (HTML'e sayı hard-code etme).
- **Kısa özet:** `lpacr/analysis/V5_ANALIZ_BULGULARI.md` — 1–2 sayfa, `HISTORY.md`'ye
  §12 olarak eklenebilecek formatta.

### 6.3 Kalite kuralları

- **Her sayı bir null ile birlikte.** Null yoksa metrik yok.
- **Belirsizlik zorunlu:** güven aralığı ya da bootstrap dağılımı olmadan nokta tahmini
  raporlama.
- **Çoklu karşılaştırma:** 84 hücre × 40 stage × 6 koşu üzerinde tarama yapıyorsun.
  Keşifsel testleri açıkça keşifsel diye etiketle; ana iddialar için FDR düzeltmesi uygula.
- **Negatif sonuç birinci sınıf sonuçtur.** "LP ölçülemiyor ve uniform doğru cevap"
  meşru ve muhtemelen doğru bir bulgu — savunmaya geçme, güçlü kanıtla yaz.
- **Uydurma yok.** Eksik dosyayı varmış gibi gösterme; markdown'daki sayıları ham veriden
  geldi gibi sunma (kaynağını "kayıtlı tablo" diye işaretle).
- **Tek seed uyarısı her yerde:** hiçbir pair çok-seed değil. Koşular arası her
  karşılaştırmanın yanına C4'ün gürültü tabanını koy.
- Ölçek/birim tuzağı: `performance` ham stage return; `spnte_lin` tracking hatası
  (**düşük iyi**). İşaret yönlerini karıştırma.

---

## 7. Çalışma tarzı

- Önce `HISTORY.md §10–§11` ve `V5_DENEME_OZETI.md`'yi oku, sonra `lpacr/analysis/`
  kodunu oku, sonra veriyi aç. Yeniden yazmadan önce mevcut fonksiyonları kullan.
- Aralarda soru sorma; belirsizlikte **varsayımını yazıp devam et** ve varsayımı raporun
  metodoloji ekinde listele.
- Tamamlayamadığın maddeyi sessizce düşürme; raporda "yapılmadı — sebep" satırı olarak
  bırak.
- Bitirdiğinde: HTML'in yolunu, tek komutluk yeniden üretme komutunu ve **en önemli 5
  bulguyu** düz metin olarak özetle.
