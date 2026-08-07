# MoE-CTS uzmanlaşma: bizim model vs upstream checkpoint'ler — karşılaştırmalı rapor

Tarih: 2026-08-07

Kaynaklar:
- Bizim: `logs/eval/moe_gate_probe/Aug03_12-01-45_moe_cts_genesis/model_7500/results.json` (35.328 sample Genesis bankası) + `moe_cts_kurtarma_planı.md` §2.1 checkpoint trendi (7500/15000/23500).
- Upstream: `upstream_mujoco_moe_uzmanlasma_sonuclari.md` (MuJoCo closed-loop, 137k ve 164k deployment-bridge checkpoint'leri, 264 rollout).
- Makale: arXiv:2602.00678, *Toward Reliable Sim-to-Real Predictability for MoE-based Robust Quadrupedal Locomotion* (§6).

Revizyon: 2026-08-07 — §6 (makalenin kendi gerekçesi ve ablasyonu), §7 (mimari analiz) ve §8 (beklenti sıralaması) eklendi; §4.1'deki load-balance iddiası düzeltildi.

---

## 0. Tek cümlelik hüküm

**Uzmanlaşma yokluğu bizim porta özgü bir hata değil.** Upstream'in yayımlanmış, JIT-parity doğrulanmış iki checkpoint'i de sparse veya terrain-semantic routing göstermiyor; gate her iki tarafta da diffuse ve terrain bilgisi hiçbir tarafta gate'ten okunmuyor. Fark derece farkı: upstream gate biraz daha yoğun, upstream expert'leri belirgin biçimde daha ayrışmış, ve upstream mixture'ı top-1'e indirgemek yıkıcı değil — bizimkinde yıkıcı.

Dahası (§6): **makalenin kendi method gerekçesi de uzmanlaşma değil, student kapasitesi.** Uzmanlaşma iddiası yalnızca abstract'ta ve PCA görselinde var; iki bağımsız protokolde tutmuyor. Yani bu mimari, adının söylediği şeyi hiç vaat etmemiş. Açık kalan tek gerçek soru: bu koşullu kapasite, **eşit parametreli düz bir encoder'dan** iyi mi? Ne makale ne biz ölçtük (`P5`).

---

## 1. Metrik sözlüğü — hangi sayı ne ölçüyor

| Metrik | Tanım | "Uzmanlaşma var" yönü | Uyarı |
|---|---|---|---|
| **Gate entropy** | Softmax gate ağırlıklarının Shannon entropisi. `ln 8 = 2.079` = tam uniform, `0` = tam hard routing. | Düşük | Tek başına performansla ilgili değil |
| **Effective experts** | `exp(entropy)`. "Kaç expert fiilen kullanılıyor" sayısı. `8` = hepsi eşit, `1` = tek expert. | Düşük | Load-balance loss'u bunu **doğrudan** yukarı itmez (yalnız batch ortalamasını kısıtlar) — bkz. §7.2 |
| **Mean max gate** | Adım başına en büyük gate ağırlığının ortalaması. Uniform = `0.125`. | Yüksek | `1/8`'e yakınlık = routing yok |
| **Marginal usage** | Her expert'in dataset ortalaması gate payı. Uniform = `0.125` her biri. | Dengesiz olması *tek başına* iyi değil | Ölü expert de dengesizlik üretir |
| **Norm-ağırlıklı effective experts (`w_eff`)** | Gate ağırlığı × expert latent normu ile yeniden hesaplanmış effective expert. | Ham `g`'den belirgin düşükse "gizli norm ekseni üzerinden routing" var demektir | Yalnız bizde ölçüldü |
| **Expert functional cosine (off-diagonal)** | Aynı state'te farklı expert'lerin latent çıktıları arasındaki ortalama kosinüs. | Düşük (0'a yakın = farklı fonksiyonlar) | Norm ölçeğinden bağımsız; asıl "kopya mı?" testi bu |
| **Param head cosine** | Expert head *ağırlıklarının* kosinüsü. | Düşük | Yüksek functional + düşük param = fonksiyonel benzerlik parametrik değil, girdi kaynaklı |
| **Single-expert action MSE (off-diagonal)** | Her expert'i tek başına latent kaynağı yapıp actor'dan geçirince çıkan action'ların çift-çift farkı. | Yüksek | Expert'ler *davranışsal* olarak farklı mı? |
| **Action gap / intervention MSE** | Gate'e müdahale edip (uniform/shuffled/top1/oracle) aynı state'te action'ın ne kadar değiştiği. | — | **Performans metriği DEĞİL.** Sadece "gate çıktıyı gerçekten etkiliyor mu?" |
| **Balanced accuracy (group-stratified)** | Gate veya latent'ten command/terrain sınıfını tahmin, rollout grupları train/test'e ayrılmış halde. | Majority baseline'ın *üzerinde* olması | Majority baseline'la birlikte okunmazsa anlamsız |
| **Normalized MI (argmax expert ↔ etiket)** | En baskın expert ile terrain/command etiketinin normalize karşılıklı bilgisi. `0` = bağımsız, `1` = birebir. | Yüksek | Bizim tarafın classifier muadili |
| **Fall rate / survival / tracking error** | Closed-loop MuJoCo: düşme oranı, ayakta kalma süresi, `‖[vx,vy,wz] − command‖` ortalaması. | — | **Tek gerçek görev metriği.** Yalnız upstream'de ölçüldü |

Kritik ayrım: **entropy/cosine/action-MSE = temsil metrikleri; fall/tracking = görev metrikleri.** Birincisinden ikincisi çıkarılamaz.

---

## 2. Yan yana ana tablo

| Ölçü | Bizim Genesis 23.5k | Upstream 137k | Upstream 164k |
|---|---:|---:|---:|
| Effective experts (ham gate) | **7.68** / 8 | 6.53 | 6.62 |
| Gate entropy | ~2.00 (@7.5k) | 1.831 | 1.851 |
| Mean max gate | **0.180** | 0.288 | 0.291 |
| Uniform referansı | 0.125 | 0.125 | 0.125 |
| Expert functional off-diag cosine | **0.278** (@7.5k) | 0.071 | 0.108 |
| Param head off-diag cosine | 0.064 (@7.5k) | ölçülmedi | ölçülmedi |
| Off-diag single-expert action MSE | ölçülmedi | 1.711 | 1.097 |
| Terrain ↔ routing ilişkisi | normalized MI **0.013** | balanced acc 0.452 < majority 0.517 | 0.479 < majority 0.538 |
| Command ↔ routing ilişkisi | vx MI **0.064** (en yüksek eksen) | gate 0.285 vs majority 0.240 | gate 0.337 vs majority 0.224 |
| Command ↔ mixed latent | ölçülmedi | **0.701** vs majority 0.220 | **0.790** vs majority 0.213 |
| Closed-loop görev metriği | **yok** | var | var |

### 2.1 Bizim checkpoint trendimiz

| Checkpoint | Effective experts | Mean max gate | Learned gap | Uniform gap | Top-1 gap | Oracle gap |
|---|---:|---:|---:|---:|---:|---:|
| 7500 | 7.46 | 0.209 | 0.472 | 0.752 | 6.941 | 1.884 |
| 15000 | 7.67 | 0.182 | 0.542 | 0.764 | 8.321 | 2.055 |
| 23500 | 7.68 | 0.180 | 0.565 | 0.774 | **7.869** | 1.894 |

Yön: eğitim ilerledikçe gate **daha uniform** hale geliyor, top-1'e indirgeme **daha da yıkıcı** oluyor.

Expert çeşitliliği trendi (functional cosine, düşmesi = çeşitlenme):
`0.603 (500) → 0.519 (2500) → 0.367 (5000) → 0.278 (7500)`

Yani expert'lerimiz eğitim boyunca ayrışmaya devam ediyordu; 23.5k'da kestik.

---

## 3. Karşılaştırılabilirlik uyarıları — bunları atlamayın

Bu iki sonuç kümesi **aynı deney değil**. Rapordaki her farkı okurken şunlar geçerli:

1. **Simulator farklı.** Bizim: Genesis/Newton, heightfield, 8192 env, training-fidelity probe bankası. Upstream: MuJoCo, exact `flat.xml` + `stairs.xml`, tek env, 5 s rollout. State dağılımları farklı.
2. **Eğitim uzunluğu 6–7 kat farklı.** 23.5k iterasyon vs 137k/164k. Bizim cosine trendimiz hâlâ düşüyordu — `0.278 vs 0.071` farkının ne kadarı "yanlış implementasyon", ne kadarı "yeterince eğitilmedi" ayrılmadı. **Bu şu an en büyük confound.**
3. **Action gap tanımları farklı.**
   - Bizim: `sum_dim((action(z_variant) − action(teacher_latent))²)`, 12 boyut üzerinden **toplam**, referans **teacher action**.
   - Upstream: `mean((action_learned − action_variant)²)`, **eleman başına ortalama**, referans **learned action** (teacher yok — bridge'de teacher/critic rastgele).
   - Kaba dönüşüm için bizimkini 12'ye bölmek gerekir, ama referans yine de farklı kalır. **Absolute değerler karşılaştırılamaz.**
4. **Terrain/command ölçüm aracı farklı.** Bizde normalized MI (argmax expert ↔ etiket), upstream'de group-stratified nearest-centroid balanced accuracy. İkisi de aynı soruyu soruyor ama aynı ölçek değil.
5. **Görev metriği asimetrik.** Upstream'de fall/survival/tracking var, bizde yok. Bizim tarafta hiçbir routing müdahalesinin *davranışsal* bedeli ölçülmedi.
6. **Tek seed, tek artifact.** Her iki tarafta da varyans ölçülmedi.

---

## 4. Analiz — dört bulgu

### 4.1 Diffuse gate ortak; bu bir port hatası değil

Upstream 137k/164k, `mean max gate ≈ 0.29` ve `effective experts ≈ 6.5–6.6`. Uniform 0.125 olduğuna göre bu "hiç routing yok" değil ama **hard/sparse routing da değil** — 8 expert'in çoğu her adımda anlamlı ağırlık alıyor. Bizim `0.180 / 7.68` değerimiz aynı rejimin daha uç ucu.

Sonuç: `go2_moects` gate'inin uniforma yakın olması **tek başına** yanlış port kanıtı değil.

> **Düzeltme (bu raporun ilk sürümüne göre).** İlk sürümde "load-balance loss'u tam olarak bunu üretiyor" yazıyordu; bu yanlıştı. [`ppo_moe_cts.py:625`](rsl_rl/algorithms/ppo_moe_cts.py:625) yalnızca **batch ortalaması** kullanımı `1/8`'e çekiyor — dengeli atama yapan mükemmel hard bir router da bu loss'u sıfırlar. Yani load-balance per-sample diffuse'luğun *sebebi* değil; sadece collapse baskısını kaldırıyor. Gerçek sebep mimari/objective geometrisi: bkz. §7.

Ek kanıt: 164k checkpoint, 137k'dan **daha diffuse** (`6.62 > 6.53`). Yani 27k iterasyon daha eğitmek gate'i keskinleştirmiyor. Bizim "23.5k'da daha eğitseydik sparse olurdu" hipotezi bu veriyle **desteklenmiyor** — gate diffuse kalıyor.

### 4.2 Terrain-semantic uzmanlaşma her iki tarafta da yok

Bu en net ortak sonuç:

- Upstream: gate ve mixed latent, terrain sınıflandırmada held-out **majority baseline'ın altında** (0.452 vs 0.517; 0.479 vs 0.538). Yani terrain bilgisi taşımıyorlar.
- Bizde: argmax expert ↔ terrain_id normalized MI = **0.0132**; terrain_level MI = 0.0042; gate_argmax ↔ terrain_id MI = 0.0023. Hepsi sıfıra yapışık.

Buna karşılık makaledeki latent PCA görselinde terrain'e göre ayrışma görünüyor (bizim replikasyonumuzda da: `terrain_pca` explained variance 0.351+0.182 = 0.533). **Bu ikisi çelişmiyor — PCA görsel ayrışması sınıflandırılabilirlik demek değil.** İki bağımsız protokol (upstream held-out classifier, bizim MI) "terrain expert" iddiasını desteklemiyor.

### 4.3 Taşınan eksen terrain değil, command

Her iki tarafta da aynı sıralama:

- Upstream: gate command'de majority'nin biraz üstünde (0.285 vs 0.240; 0.337 vs 0.224), **mixed latent command'de çok güçlü** (0.701 ve 0.790 vs ~0.22 majority).
- Bizde: `vx_bin` MI = 0.0641, terrain MI'ın ~5 katı ve tüm eksenlerin en yükseği (`yaw_bin` 0.0024, `terrain_level` 0.0042).

Yorum: bu ağ terrain uzmanları değil, **command-koşullu bir latent** öğreniyor. `H4` hipotezi (kurtarma planı §2.2) — "terrain semantiği history'den yeterince gözlenebilir değil, command/contact/phase daha gerçekçi hedef" — upstream verisiyle **doğrulandı**.

Ayrıca: bilgi ağırlıklı olarak **gate'te değil mixed latent'te**. Upstream'de gate 0.29–0.34, mixed latent 0.70–0.79. Yani routing kararı değil, karışımın kendisi taşıyor bilgiyi. Bu, "MoE burada conditional ensemble / dense basis encoder gibi çalışıyor" okumasını destekliyor — ki kurtarma planımızın §1'deki teşhisi zaten buydu.

### 4.4 Tek gerçek nitel fark: top-1 dayanıklılığı

Bu, iki model arasındaki en anlamlı ayrım:

| | Bizim 23.5k | Upstream 137k | Upstream 164k |
|---|---|---|---|
| Top-1'e indirgeme etkisi | learned gap 0.565 → top-1 gap **7.869** (≈14×) | learned–top1 MSE 0.383 | learned–top1 MSE 0.223 |
| Top-1 closed-loop | ölçülmedi | fall 0.333 (learned ile aynı), tracking 0.623 vs 0.334 | fall **0.000**, tracking 0.566 vs 0.386 |

Upstream'de top-1 routing **çalışıyor** — 164k'da hiç düşmüyor, sadece tracking'i kötüleşiyor. Bizde top-1 latent'i tanınmaz hale getiriyor (`latent_cos 0.480`, action gap 14×).

Bunun anlamı: upstream mixture'da her expert tek başına da makul bir latent üretiyor; karışım **redundant**. Bizimkinde karışım **hassas dengeli bir toplam** — tek bileşen anlamsız. Bu, expert cosine farkıyla (0.278 vs 0.071) da tutarlı: onların expert'leri ortogonale yakın ve her biri kendi başına ayakta, bizimkiler birbirini tamamlayan ama tek başına yetersiz bileşenler.

Ancak dikkat: bu karşılaştırma §3.3'teki tanım farkı yüzünden **kesin değil**. Kesinleştirmek için bizim bankada `learned` ile `top1` action'ları arasındaki **pairwise per-element MSE**'yi ölçmek gerekiyor — ucuz, mevcut probe'ta tek satır.

---

## 5. Upstream'in kendi içinde: 137k → 164k ne değişti

| Eksen | Yön |
|---|---|
| Gate concentration | Biraz **daha diffuse** (6.53 → 6.62) — daha uzun eğitim gate'i keskinleştirmiyor |
| Command decoding | **İyileşti** (gate 0.285→0.337, latent 0.701→0.790) |
| Terrain decoding | Değişmedi, ikisi de majority altında |
| Expert functional diversity | **Azaldı** (cosine 0.071→0.108, action MSE 1.711→1.097) |
| Closed-loop | 164k'da top-1 fall rate `%0`; learned tracking biraz daha kötü (0.386 vs 0.334) |

Yani upstream'in kendi eğitim gidişatı da "daha uzun eğitim → daha sparse, daha uzmanlaşmış expert" demiyor. Tersine: command bilgisi artıyor, expert çeşitliliği azalıyor, gate diffuse kalıyor.

---

## 6. Makale MoE'yi neden koydu — kendi beyanı ve kendi ablasyonu

Kaynak: arXiv:2602.00678, *"Toward Reliable Sim-to-Real Predictability for MoE-based Robust Quadrupedal Locomotion"*.

### 6.1 Beyan edilen gerekçe: uzmanlaşma değil, **öğrenci kapasitesi**

Method bölümünün gerekçesi:

> "The limited expressive capacity of the student model often precludes it from accurately inferring the features encoded by the teacher, which consequently restricts the performance ceiling."

Yani MoE burada bir **distilasyon darboğazı yaması**: CTS'te student (proprioseptif geçmiş) teacher'ın (privileged) latent'ini taklit etmek zorunda, student yeterince ifade gücüne sahip olmayınca tavan düşük kalıyor. Çözüm olarak student encoder'ın yerine MoE konuyor.

Buna karşılık **abstract farklı bir hikâye anlatıyor:**

> "The MoE policy employs a gated set of specialist experts to decompose latent terrain and command modeling…"

İki anlatı yan yana duruyor: *method'da kapasite, abstract'ta semantik ayrıştırma.* §4.2–4.3'teki ölçümler — hem bizimkiler hem upstream checkpoint'lerininki — **birinciyi destekliyor, ikinciyi desteklemiyor.**

Ek işaret: makale Switch Transformer ve Sparsely-Gated MoE'yi kaynak gösteriyor ama **top-k'yı almıyor**. Gating formülasyonu düz dense softmax:

```
z_s = Σ_k ω_k E_k(o_t),      ω_k = softmax(g(o_t))_k
L_load_balance = Σ_k (ω̄_k − 1/K)²,   ω̄_k = (1/B) Σ_j ω_k^(j)
```

Yani çerçeve sparse MoE literatüründen alınmış, mekanizma alınmamış. Load-balance'ın batch-ortalaması üzerinde tanımlı olduğu makalenin kendi denkleminde de açık.

### 6.2 Makalenin ablasyon tablosu

| Model | Score | Tracking | Safety | Quality | Level |
|---|---:|---:|---:|---:|---:|
| **MoE (Ours)** | **0.6745** | 0.6574 | 0.7765 | 0.7722 | 7.81 |
| MoE-NG | 0.6637 | 0.6450 | 0.7651 | 0.7615 | 7.67 |
| AC-MoE | 0.6589 | 0.6402 | 0.7601 | 0.7538 | 7.56 |
| MCP | 0.6513 | 0.6343 | 0.7559 | 0.7504 | 7.52 |

Tanımlar (makaleden):
- **MoE-NG**: "The command information is excluded from the MoE input, utilizing only observation information to the expert networks."
- **AC-MoE**: "Following MoE-Loco, the MoE structure is applied to the Actor-Critic networks rather than the student encoder."
- **MCP**: "A multiplicative composition strategy is employed for the actions output by the Actor."

Ana karşılaştırma tablosu (Table IV): `MoE 0.67 / CTS 0.58 / HIM 0.53 / DreamWaQ 0.47`.

### 6.3 Bu tablodan çıkan iki kritik sonuç

**(a) Koşullandırmanın getirisi %1.6.** `MoE → MoE-NG` farkı `0.0108 / 0.6745 = %1.6`. MoE-NG, komutu MoE girdisinden çıkarıyor — yani gate'in/expert'lerin komuta koşullanmasının tüm katkısı bu. Bu, bizim MI ölçümümüzle bağımsız olarak örtüşüyor: `vx_bin` MI = `0.064` en yüksek eksendi ama mutlak olarak küçüktü (§4.3). İki farklı yöntem aynı şeyi söylüyor — **gate komutu biraz taşıyor, terrain'i hiç taşımıyor, ve etkinin büyüklüğü küçük.**

**(b) Ablasyonda düz MLP yok, parametre eşitlemesi hiç yok.** Ablasyondaki dört satırın hepsi MoE varyantı. MoE'siz tek kıyas ana tablodaki `CTS 0.58`. Ama `CTS → MoE` geçişinde student ağı da büyüyor (§7.1: penultimate katman 8× geniş) ve makalede **parametre eşitlendiğine dair hiçbir ifade yok**. Dolayısıyla `+0.09`'un ne kadarı "MoE mekanizması", ne kadarı "daha büyük student" — makale bunu ayırmıyor.

Bu, `H5` hipotezinin ve `P5` adımının tam olarak kapatacağı delik. Makale bu ölçümü yapmamış; biz yaparsak literatürde olmayan bir sayı üretmiş oluruz.

---

## 7. Mimari: diffuse gate neden bu objective'in **optimumu**

### 7.1 "8 expert" aslında ne

[`moe_utils.py:69-120`](rsl_rl/modules/moe_utils.py:69) ve upstream'in birebir aynısı [`go2_rl_gym/rsl_rl/rsl_rl/modules/utils.py:69,87,96,119`](go2_rl_gym/rsl_rl/rsl_rl/modules/utils.py:69):

```
h(x)   = backbone(x)                          # 225 → 512 → 256 → 2048, elu  (PAYLAŞILAN)
E_i(x) = A_i · h_i(x)                         # h_i = h'nin i. 256'lık dilimi, A_i: 256→32
latent = L2Norm( Σ_i w_i(x) · A_i · h_i(x) )  # w = softmax(gate_mlp(x)), top-k YOK
```

Grouped `nn.Conv1d(2048 → 8·32, groups=8)` ile uygulanıyor. Yani:

- **Tüm nonlineer hesap paylaşılıyor.** 8 "expert" = ortak gövdenin üstündeki 8 tane *tek katmanlı lineer okuma başlığı*.
- Bunlar 8 politika, 8 beceri, 8 yürüyüş tarzı değil — 32 boyutlu latent uzayında **8 taban vektörü**.
- MoE yalnızca student history encoder'da; actor `[latent(32), obs(45)]` üzerinde düz bir MLP.

Tek satırda: `latent = L2Norm( A(x) · h(x) )`, burada `A(x)` blokları `[w₁(x)A₁, …, w₈(x)A₈]` olan 32×2048 bir matris. **Efektif son katman girdiye göre değişiyor — ama sadece 8 skalerlik serbestlikle.** Buna "koşullu ekstra kapasite" diyoruz: FiLM/gating ailesinden, hypernetwork'ün çok kısıtlı bir hâli.

### 7.2 Neden diffuse

Objective: `MSE( L2Norm(Σ w_i E_i), teacher_latent )`. Sabit bir taban kümesinden bir hedef vektörü L2 anlamında en iyi yaklaşıklamak **genel olarak yoğun** bir kombinasyon ister; tek bir taban elemanı seçmek neredeyse her zaman daha kötü çözümdür. Üstelik çıktı birim küreye normalize ediliyor — R³²'de kürenin rastgele bir yönüne ulaşmak için karıştırmak zorunlu.

Ve objective'de uzmanlaşmayı ödüllendiren **hiçbir terim yok**:
- top-k yok → seyrek seçim zorunluluğu yok
- sparsity/entropy cezası yok, temperature yok
- load-balance yalnız batch ortalamasını kısıtlıyor (§4.1 düzeltmesi)

Bu yüzden `top1` bizim modelde latent'i tanınmaz hale getiriyor (`latent_cos 0.480`) — sparse routing bu objective'e göre **daha kötü** bir çözüm. Model diffuse olmayı "seçmiyor"; diffuse olmak çözümün kendisi.

**Dolayısıyla:** "uzmanlaşma yok" bulgusu ne bizim implementasyon hatamız ne de eğitim eksikliğimiz. Yöntemde uzmanlaşma üreten bir sinyal yok. Upstream'in 137k/164k checkpoint'lerinde de olmaması bunun doğrudan kanıtı.

---

## 8. Beklenti sıralaması: uzmanlaşmış MoE vs bizimki vs eşit-parametreli MLP

Bu bölüm **ölçüm değil, gerekçelendirilmiş beklenti.** Ölçüm `P5`.

Beklenen sıralama:

> **mevcut soft MoE  ≳  eşit-parametreli düz MLP  >  hard/top-k uzmanlaşmış MoE**

**Soft MoE vs eşit-param MLP — küçük farkla MoE.** Girdiye koşullu son katman, sabit ağırlıklı MLP'nin sahip olmadığı çarpımsal bir etkileşim ekliyor; gated mimariler eşit parametrede tipik olarak biraz önde olur. Ama fark küçük olmalı, çünkü modülasyon bant genişliği 8 skaler. Üç bağımsız işaret bunu destekliyor: bizde uniform gate bozuyor ama yıkmıyor (`0.565 → 0.774`), upstream'de uniform closed-loop'ta ayakta kalıyor, makalenin kendi MoE-NG ablasyonu %1.6.

**Uzmanlaşmış (hard/top-k) MoE bu problemde neden muhtemelen daha kötü:**

1. **Hedef sürekli.** Terrain ve komut sürekli değişkenler (eğim açısı, basamak yüksekliği, hız). Sürekli bir manifoldu sert bölmek sınırlarda süreksizlik yaratır — 50 Hz'de adım ortasında zemin değişirken kontrolcü için kötü. Yoğun karışım pürüzsüz interpolasyon veriyor.
2. **Rejimler ağır örtüşüyor.** "0.5 m/s'te eğimde yürüme" ile "0.5 m/s'te düzde yürüme" neredeyse her şeyi paylaşıyor. Uzmanlaşmanın kazandırdığı yer alt-görevlerin *ayrık* olduğu yerdir; bu görev setinde değiller.
3. **Örneklem verimliliği.** Her expert verinin ~1/K'sını görür. RL'de veri dağılımı durağan değil (terrain curriculum!) — bir expert aç kalıp sonra aniden gerekli hale gelebilir.
4. **Kararlılık.** Seyrek routing'i RL ile birlikte eğitmek kırılgan. Makalenin kendi AC-MoE/MCP ablasyonları için yazdığı ifade: "prone to loss divergence."

**Uzmanlaşmış MoE nerede kazanırdı:** gerçekten ayrık gait rejimleri olsaydı (yürüme vs boşluk üstünden atlama vs sürünme) ve açık rejim etiketi/supervision olsaydı.

**Düzeltilmesi gereken varsayım:** "uzmanlaşmış MoE = ideal MoE" bu problem için doğru değil. Bu görevin ideal MoE'si muhtemelen zaten yoğun olanı. Bulduğumuz şey bir arıza değil — mimarinin doğru çözümü; sorun mimarinin **adının** yanlış olması.

---

## 9. Bizim implementasyon için ne demek

**Ortadan kalkan endişeler:**
- "Gate'imiz uniform, port yanlış" → hayır, upstream de diffuse.
- "Terrain expert çıkmadı, bir şeyi kaçırdık" → hayır, upstream'de de yok.
- "Load-balance katsayısı yanlış" → iki tarafta da `0.01`, iki tarafta da aynı sonuç.

**Ayakta kalan gerçek farklar (öncelik sırasıyla):**

1. **Eğitim uzunluğu.** 23.5k vs 137k. Bizim expert cosine trendimiz kesildiği anda hâlâ düşüyordu (0.60→0.28). Bu, `0.278 vs 0.071` farkının en olası açıklaması ve **düzeltilebilir olan tek şey** — ama gate concentration'ı düzeltmeyeceği 164k verisinden belli.
2. **Top-1 kırılganlığı.** Bizim mixture çok daha kırılgan. Bu ya (1)'in sonucu, ya da gerçek bir temsil farkı. Ayırt etmek için önce ölçüm tanımını eşitlemek gerekiyor.
3. **Closed-loop kanıt yokluğu.** Bizim tarafta hiçbir routing müdahalesinin görev bedeli ölçülmedi. Upstream'de learned route uniform/top-1'den daha iyi tracking veriyor — yani routing'in *gerçek* bir davranışsal değeri var. Bizimkinde bunu bilmiyoruz.

**Planla ilişki:** kurtarma planındaki `H1` (mixed loss local kredi vermiyor → basis bileşenleri) ve `H4` (terrain gözlenebilir değil, command daha gerçekçi hedef) upstream verisiyle destekleniyor. `H5` (avantaj routing'den değil kapasiteden) hâlâ açık — ve §6.1'de görüldüğü gibi **makalenin kendi method gerekçesi de H5'i söylüyor.** `P5` parameter-matched dense baseline'ı artık planın **en yüksek öncelikli** adımı: makalede olmayan, literatürde eksik olan tek sayı bu.

---

## 10. Sıradaki en ucuz ayırt edici ölçümler

Ucuzdan pahalıya:

1. **Bizim bankada pairwise action MSE** (`learned` vs `uniform`/`top1`/`shuffled`, per-element mean). Upstream'le doğrudan karşılaştırılabilir hale getirir. Maliyet: mevcut `probe_moe_gate.py` içinde birkaç satır, yeni rollout yok.
2. **Bizim modelde single-expert action MSE off-diagonal.** Upstream'in 1.711/1.097 değerinin muadili. Aynı banka, yeni rollout yok.
3. **Bizim modeli aynı MuJoCo protokolüne sokmak** — mümkün değilse Genesis'te aynı 11 route (learned/uniform/top1/fixed_expert_0..7) × command × terrain closed-loop kampanyası. Fall/survival/tracking olmadan "routing işe yarıyor mu" sorusu cevapsız kalıyor. **En yüksek bilgi/maliyet oranı bu.**
4. **Group-stratified command/terrain classifier'ı bizim gate ve mixed latent'imize uygulamak.** MI yerine upstream'le aynı ölçek. Upstream'in `0.70–0.79` mixed-latent command accuracy'sine karşı bizim değerimiz, "aynı bilgiyi mi taşıyoruz" sorusunun doğrudan cevabı.
5. **23.5k → daha uzun eğitim, 3 seed.** Pahalı; ancak (1)–(4) sonrası mantıklı.

Terrain-semantic routing gerçekten isteniyorsa: mevcut objective'in bunu üretmediği artık iki bağımsız modelde gösterildi. Explicit terrain/regime supervision veya contrastive routing objective gerekiyor — bu bir hiperparametre ayarı değil, objective değişikliği.

---

## 11. Ne söylenemez

- "Bizim implementasyon upstream'e denk" — denk değil; eğitim uzunluğu, top-1 dayanıklılığı ve closed-loop kanıtı farklı.
- "Upstream de başarısız" — upstream'in learned route'u closed-loop'ta uniform ve fixed-expert'lerden daha iyi tracking veriyor; routing'in ölçülmüş davranışsal değeri var. Bizde ölçülmedi.
- "Terrain uzmanlaşması makalede yok" — makale iddiası bu bankada desteklenmedi; upstream'in tam eğitim terrain bankasında değil, `flat`+`stairs` üzerinde ölçüldü.
- "Action MSE'ler performans" — değil, aynı-state müdahale ölçüleridir.
- "MoE gereksiz / MLP yeterli" — bu **ölçülmedi**. §8 bir beklenti, sonuç değil. `P5` yapılmadan söylenemez.
- "Makale yanlış" — makalenin ana tablosu (`MoE 0.67 vs CTS 0.58`) gerçek bir kazanç gösteriyor; gösteremediği şey o kazancın **kaynağı** (routing mi, kapasite mi). Eleştiri sonuçlara değil atıf yapılan mekanizmaya.
