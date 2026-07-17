# V3 Genelleme Değerlendirme Planı

## 1. Amaç ve ön-kayıt sınırı

Bu belge, V3 eğitim kampanyası için **eğitim tamamlanmadan ve sonuçlar görülmeden önce** sabitlenen değerlendirme sözleşmesidir. Amaç, bir metodun yalnız ham tracking değeriyle iyi görünüp görünmediğini değil, aşağıdaki üç soruya cevap vermektir:

1. Online adaptasyon, in-distribution (ID) kontrol kalitesine bedel yüklüyor mu?
2. Statik fakat alışılmadık payload/fizik koşullarında, MLP ile ulaşılabilir oracle tavanı arasındaki boşluğun ne kadarını kapatıyor?
3. Fizik episode ortasında değiştiğinde, ne kadar hızlı ve güvenli toparlıyor?

Bu planın başlık sonuçları yalnız aşağıda tanımlanan **headline-eligible** hücrelerden hesaplanır. Geçerlilik kapısından geçmeyen hücreler silinmez; ham sonuçları, kapı durumu ve geçememe nedeni kalıcı artefact olarak saklanır ve tanısal analizde kullanılır.

## 2. Donmuş V3 eğitim sözleşmesi

Bu değerlendirme yalnız hâlihazırda eğitilen V3 ailesinin gerçekten gördüğü dağılım ve gözlem sözleşmesi üzerinde ana iddia kurar.

| Başlık | V3 sözleşmesi |
|---|---|
| Yöntemler | `go2_v3_mlp`, `go2_v3_sysid`, `go2_v3_rma`, `go2_v3_dreamwaq`, `go2_v3_him_fixed`, `go2_v3_superset_oracle` |
| Training seed | Mevcut kampanyada method başına seed 1 ve seed 2 (`N_seed=2`) |
| Mass bandı | `added_mass ∈ [-2, +5] kg` |
| CoM bandı | `com_x`, `com_y`, `com_z ∈ [-0.08, +0.08] m` |
| Episode içi değişim | En fazla bir mass+CoM resample; episode uzunluğunun `%20–80` aralığında |
| Forward command zarfı | Curriculum sonunda `lin_vel_x ∈ [-1, +2] m/s` |
| Lateral / yaw command zarfı | `lin_vel_y, ang_vel_yaw ∈ [-1, +1]`; V3 bunları genişletmez |
| SysID hedefi | Gerçek base velocity (`3`) + `P5=[friction, mass, com_x, com_y, com_z]` |
| Superset-Oracle | 20-frame proprio history + gerçek velocity + gerçek `P5` |

`pd_gain` V3 eğitiminde randomize edilmemiştir ve `P5`in parçası değildir. Bu nedenle PD-gain sonuçları, varsa, **held-out diagnostic** olarak raporlanır; V3 adaptasyon headline skoruna dahil edilmez. Terrain için de oracle height scan taşımadığından, terrain hücreleri `gap_closed` hesabına katılmaz; ham dayanıklılık sonucu olarak tutulur.

## 3. Model, checkpoint ve tekrar birimi

### 3.1 Model matrisi

Her suite, seçilmiş checkpoint ile aşağıdaki method × training-seed matrisi üzerinde çalışır:

| Etiket | Task | Seed |
|---|---|---|
| MLP | `go2_v3_mlp` | 1, 2 |
| SysID | `go2_v3_sysid` | 1, 2 |
| RMA | `go2_v3_rma` | 1, 2 |
| DreamWaQ | `go2_v3_dreamwaq` | 1, 2 |
| HIM-fixed | `go2_v3_him_fixed` | 1, 2 |
| Superset-Oracle | `go2_v3_superset_oracle` | 1, 2 |

Checkpoint olarak training-internal deterministic validation ile seçilen `best_tracking.pt` kullanılır. Başka checkpoint ancak aynı validation protokolüyle yeniden seçilmiş ve manifest’e açıkça yazılmışsa kullanılabilir. En son `model_3000.pt` otomatik olarak en iyi checkpoint kabul edilmez.

### 3.2 Tekrar birimi

Bilimsel tekrar birimi environment değil, **training seed**dir. Bir hücredeki paralel environment’lar ölçüm hassasiyeti sağlar; bağımsız policy eğitimi sağlamaz.

Bu kampanya için önceden sabitlenmiş kural:

```text
N_seed = 2
pozitif işaret tutarlılığı = iki training seed'in ikisinde de gap_closed > 0
```

İleride üçüncü seed eklenirse aynı kural genellenir: önceden ilan edilmiş bütün training seed’lerde `gap_closed > 0` olmalıdır. Mevcut iki seed sonucu sonradan üç-seed iddiası gibi gösterilmez.

## 4. Ortak ölçüm protokolü

| Ayar | Değer |
|---|---:|
| Paralel environment / nihai hücre | `384` |
| Ölçülen adım / statik hücre | `2000` |
| Warmup | `100` adım; suite özelinde ayrıca belirtilmedikçe |
| Physics/command scenario bank | Deterministik, methodlar arasında aynı |
| Ölçüm gürültüsü | Açık; eğitim/deploy gözlem sözleşmesi korunur |
| Checkpoint | `best_tracking.pt` + SHA-256 manifest |

`384 × 2000 = 768.000` environment-step, tek hücredeki tracking/fall ölçümü için yeterli hassasiyet sağlar. 768 yerine 384 seçimi, asıl bilimsel tekrar olan training seed sayısını değiştirmeden eval maliyetini azaltır.

Her hücrede en az şu ham metrikler saklanır:

```text
tracking_lin_err
tracking_ang_err
achieved_speed
achieved_speed_ratio
fall_rate
return_per_step
pre_switch_tracking_error          # switch hücrelerinde
post_switch_tracking_error         # switch hücrelerinde
post_switch_tracking_yaw           # switch hücrelerinde
post_switch_fall_rate              # transient/switch hücrelerinde
```

SysID için switch senaryolarında ayrıca zaman serisi saklanır:

```text
P_true(t)
P_hat(t)
mean_error(t)
MAE(t)
RMSE(t)
```

## 5. Suite S0 — ID paritesi

S0’ın sorusu: **Adaptasyonun normal çalışmadaki bedeli var mı?**

S0 iki alt rapora ayrılır; ikisi birbiriyle karıştırılmaz.

### S0a — Static ID

- Fizik episode boyunca sabittir.
- Mass ve CoM V3 eğitim bandından örneklenir.
- Deterministik command bank kapsanır: `vx∈[-1,2]`, `vy∈[-1,1]`, `yaw_rate∈[-1,1]`.
- Eval `heading_command=False` ve command resampling kapalıdır; üçüncü command bileşeni doğrudan yaw-rate’tir, heading hedefi değildir.
- Ana rapor: lineer/yaw tracking, fall ve MLP’ye göre fark.

Manşet sayı:

```text
ID_Δ = static-ID'de method ile MLP arasındaki tracking/fall farkı
```

Beklenti `ID_Δ ≈ 0`dır: estimator normal koşullarda anlamlı bir tracking veya stability bedeli yaratmamalıdır.

### S0b — Dynamic ID

- Physics switch, V3’te eğitilen in-band mass+CoM sözleşmesine uyar.
- Bu bölüm headline `ID_Δ` ile birleştirilmez.
- Amaç, policy’nin eğitimde gördüğü switch becerisinin sağlıklı çalıştığını doğrulamaktır.

## 6. Suite S1 — Statik payload OOD

S1’in sorusu: **Fizik sabit ama training bandının sınırında/dışındayken online adaptasyon oracle boşluğunu kapatıyor mu?**

### 6.1 Birincil fizik senaryosu: payload-composite

Ana eksen saf mass değildir. Her severity noktası, gerçekçi bir yükü temsil eden deterministik bir bileşik physics setter ile uygulanır:

```text
payload(s) = added_mass(s) + com_x(s)
```

`com_x(s)` fonksiyonu, sign, maksimum deplasman ve simulator geri-okuma doğrulaması implementation öncesinde ayrı bir fizik tablosunda sabitlenir. Aynı payload senaryosu bütün yöntemlerde, seed’lerde ve tekrar koşularında bire bir uygulanır.

### 6.2 Severity katmanları

| Katman | Tanım |
|---|---|
| `in_band` | V3 mass bandı içi: `[-2, +5] kg` |
| `near_ood` | Bandın hemen dışı; başlangıç adayı `+6 kg`, gerekirse negatif yönde `-3 kg` |
| `far_ood` | Daha dış payload; başlangıç adayları `+7`, `+8 kg` |

Final grid, payload-composite mapping’i ve simulator stabilite smoke’u ardından bu belgeye eklenir; sonuç görüldükten sonra değiştirilemez.

### 6.3 Komut hücreleri

Birincil command set:

```text
lateral: -1.0, -0.5, +0.5, +1.0 m/s
forward: 1.5, 2.0 m/s
```

Lateral testinde `vx=0` ve `yaw_rate=0`; forward testinde `vy=0` ve `yaw_rate=0` sabitlenir. Böylece bir headline hücresinde yalnız payload/fizik değişir; command bileşeni training support dışına çıkmaz.

`|vy|∈{1.5,2.0}` lateral hücreleri `command_ood_diagnostic` olarak ayrıca koşulabilir. Ham artefact ve diagnostic atlas’ta kalırlar, fakat physics OOD ile command OOD’u ayıramadıkları için hiçbir `GapClosed_static` medyanına girmezler. Forward yüksek-hız hücreleri ancak oracle achieved-speed `%90` eşiğini geçtiğinde headline’a katılır; geçmezlerse ham raporda `oracle_speed_saturated` etiketiyle kalırlar.

### 6.4 İkincil tanısal eksenler

Saf `added_mass × forward` deneyleri, ana grid bütçesi yerine appendix/diagnostic katmanına alınır. PD gain, V3’ün tahmin uzayı dışında kaldığı için held-out diagnostic’tir; headline `GapClosed_static` hesabında kullanılmaz.

## 7. Suite S2 — Deterministik dynamic payload switch

S2 ana adaptasyon suite’idir. Soru şudur:

> Fizik yürüyüş sırasında değiştiğinde method ne kadar hızlı, ne kadar düşük hata ile ve düşmeden yeni fiziğe yeniden uyum sağlar?

### 7.1 Ortak zaman çizelgesi

Her environment aynı deterministic zaman çizelgesini görür:

```text
reset
→ warmup
→ switch öncesi sabit-physics penceresi
→ switch (t = T/2)
→ sabit uzunlukta switch sonrası ölçüm penceresi
```

Switch, estimator history’si dolduktan sonra uygulanır. Eval’de switch anı random draw’a bırakılmaz; böylece bütün methodlar aynı disturbance’ı aynı adımda görür.

### 7.2 Senaryolar

| Senaryo | Başlangıç → hedef fizik | Yorum |
|---|---|---|
| S2-A | nominal → in-band payload | Saf, V3 eğitim dağılımı içi adaptasyon |
| S2-B | nominal → near/far-OOD payload | Adaptasyon + ekstrapolasyon |
| S2-C | in-band payload A → in-band payload B | İsteğe bağlı simetrik değişim kontrolü |

Ana S2 raporu S2-A ve S2-B’yi ayrı gösterir; bunlar tek medyana karıştırılmaz. S2-A, “eğitilmiş adaptasyon kabiliyeti”; S2-B, “training support dışına taşınca dayanıklılık” olarak yorumlanır.

S2’de de aynı command ayrıştırma kuralı geçerlidir: headline lateral hücreleri `vy=±1.0` içinde tutulur, forward `vx=1.5` support içindedir ve test edilmeyen iki bileşen sıfırdır. `vy=±1.5` yalnız `command_ood_diagnostic`tır; dynamic physics headline’ına girmez.

### 7.3 Switch metrikleri

Switch sonrası sabit pencerede:

```text
post_switch_tracking_error
post_switch_tracking_yaw
post_switch_fall_rate
raw_gap_closed
headline_gap_closed
```

Ana dynamic tracking metriği, sabit uzunluktaki switch-sonrası pencerenin ortalama lineer tracking hatasıdır. Pencere uzunluğu bütün hücrelerde aynı olduğundan ayrıca integral veya eşik tabanlı `recovery_time` hesaplanmaz. Yaw hatası ve pencere içinde düşen environment oranı ayrı ham metrikler olarak raporlanır.

SysID raporu aynı zamanda `P_hat(t)` ile gerçek switch parametresinin arasındaki yakınsamayı gösterir. Environment ortalamasındaki işaret iptalini gizlememek için signed mean error yanında MAE ve RMSE zaman serileri saklanır. Bu figür tracking iyileşmesinin gerçekten estimator davranışıyla ilişkisini denetlemek içindir; tek başına başarı metriği değildir.

## 8. Suite S3 — Kalibre edilmiş hard diagnostic senaryolar

S3 headline `gap_closed` skoruna girmez. Amaç, daha zor dayanıklılık bölgelerinde ham davranışı göstermektir.

### Kick

Kick şiddeti önce MLP üzerinde hızlı bir kalibrasyon taramasıyla seçilir. Nihai seçilmiş şiddette MLP fall rate’inin yaklaşık `%30–70` bandında olması hedeflenir. Herkesin düştüğü veya hiç kimsenin düşmediği koşullar nihai S3 hücresi değildir.

### Terrain

Terrain hücreleri, oracle’ın height scan erişimi olmadığı sürece raw tracking/fall/robustness raporu olarak kalır. Oracle–MLP `gap_closed` hesabına sokulmaz.

## 9. Hücre skoru ve geçerlilik kapıları

### 9.1 Ham skor

Her method hücresi için ham skor hesaplanır:

```text
raw_gap_closed =
  (err_MLP − err_method) /
  (err_MLP − err_SupersetOracle)
```

Görselleştirme ve aykırı değerlerin suite medyanını bozmasını önlemek için yalnız yayımlanan skor değeri `[-0.5, 1.5]` aralığına kırpılır. Kırpılmamış değer ayrıca artefact’te tutulur.

Yorum:

| Değer | Yorum |
|---:|---|
| `0` | MLP’den fark yok |
| `1` | Superset-Oracle seviyesine ulaştı |
| `0.5` | MLP–oracle boşluğunun yarısını kapattı |
| `<0` | MLP’den kötü |

`err` S1 için ilgili hücrenin statik ortalama tracking hatası, S2 için sabit switch-sonrası pencerenin ortalama tracking hatası olarak tanımlanır. Bir suite içinde farklı hata tanımları medyana karıştırılmaz.

### 9.2 Oracle headroom kapısı

Bir hücre, yalnız Superset-Oracle MLP’den:

1. en az `%10` göreli tracking iyileşmesi sağlıyorsa, **ve**
2. implementation öncesi sabitlenecek asgari mutlak hata marjını geçiyorsa

headline-eligible olabilir.

Bu koşul sağlanmazsa hücrenin sorusu tanımsızdır; oracle ve MLP arasında kapanacak anlamlı boşluk yoktur.

### 9.3 Achieved-speed kapısı

Bir command hücresi yalnız oracle achieved speed’i hedef komutun en az `%90`ı ise headline-eligible olabilir:

```text
oracle_achieved_speed_ratio ≥ 0.90
```

Bu, özellikle yüksek forward komutlarında, fizik adaptasyonu yerine erişim/komut doyumunun ölçülmesini engeller.

### 9.4 Ceiling-bütünlüğü kapısı

Oracle, MLP’den tracking’de iyi görünürken stability’de anlamsız derecede kötü olmamalıdır:

```text
oracle_fall_rate ≤ mlp_fall_rate + 5 yüzde puan
```

Bu koşul sağlanmazsa oracle güvenilir bir tavan değildir ve hücre headline’a alınmaz.

### 9.5 Method fall-gate

İncelenen method MLP’den `5` yüzde puandan fazla düşüyorsa:

```text
method_fall_rate > mlp_fall_rate + 5 yüzde puan
```

ham sonuç saklanır, fakat:

```text
headline_gap_closed = 0
headline_status = method_fall_gated
```

Tracking kazancı düşme pahasına satın alınamaz.

### 9.6 Durum taksonomisi

| `headline_status` | Ham artefact | Headline agregasyonu |
|---|---:|---:|
| `eligible` | Saklanır | Dahil edilir |
| `no_oracle_headroom` | Saklanır | Dahil edilmez |
| `oracle_speed_saturated` | Saklanır | Dahil edilmez |
| `oracle_unstable` | Saklanır | Dahil edilmez |
| `method_fall_gated` | Saklanır | Dahil edilir; skor `0` |
| `diagnostic_only` | Saklanır | Dahil edilmez |

Oracle kaynaklı geçersizlikler hiçbir zaman gizli sıfıra çevrilmez. Çünkü bu hücrelerde payda bilimsel olarak anlamlı değildir. Method fall-gate ise geçerli bir karşılaştırmada güvenlik kuralı ihlalidir; headline kredi sıfırlanır.

## 10. Agregasyon ve üç manşet sayı

Agregasyon sırası:

```text
environment/replica ölçümleri
→ seed bazlı hücre metriği
→ yalnız headline-eligible hücrelerde hücre medyanı
→ seed medyanı [min, max]
```

Raporlanacak üç manşet:

| Metrik | Tanım |
|---|---|
| `ID_Δ` | S0a static-ID’de MLP’ye göre tracking/fall farkı; yaklaşık `0` beklenir |
| `GapClosed_static` | S1 headline-eligible hücrelerinin medyanı |
| `GapClosed_dynamic` | S2 headline-eligible hücrelerinin medyanı; S2-A ve S2-B ayrı verilir |

Bir method için “pozitif gap closing” iddiası, önceden tanımlı bütün training seed’lerde `gap_closed > 0` işaret tutarlılığı gerektirir. Medyan tek başına yeterli değildir.

Superset-Oracle için seed’ler arasından en iyi geçerli sonuç tavan olarak seçilebilir; method için seed medyanı kullanılır. Bu asimetri spec’te açıkça raporlanır: daha güçlü tavan, method için daha muhafazakâr oran üretir.

## 11. Artefact ve rapor sözleşmesi

### 11.1 Her ham hücre

Her çıktı aşağıdaki kimlik ve provenance alanlarını taşır:

```text
campaign
suite / scenario / severity / command
model / task / training_seed
checkpoint_path / checkpoint_sha256 / checkpoint_iteration
eval_seed
requested_physics / simulator_readback_physics
num_envs / warmup / measured_steps
```

Her hücre için raw metrikler, ham/kırpılmış skor, bütün kapı boolean’ları ve `headline_exclusion_reasons[]` yazılır. Physics setter’ın simulator’dan geri-okunan değeri, istenen scenario ile uyuşmazsa çalışma fail-loud olur.

### 11.2 Rapor katmanları

Rapor iki ayrı yüz taşır:

1. **Headline scorecard:** yalnız eligibility sözleşmesini sağlayan hücreler; üç manşet sayı, seed medyanı ve min–max bandı.
2. **Diagnostic atlas:** bütün hücreler; raw tracking/fall/achieved speed, raw gap score, kapı durumları ve geçersizlik nedenleri.

Diagnostic atlas’ın amacı kötü/sature hücreleri saklamak değil, hangi bölgenin neden ayrıştırmadığını görünür yapmaktır.

## 12. Uygulama sırası

1. Bu planı dondur; yalnız aşağıdaki açık parametreleri sonuçlar görülmeden önce doldur.
2. V3 eğitimlerinin tamamlanmasını ve her method × seed için `best_tracking.pt` oluşmasını bekle.
3. Checkpoint manifest ve SHA-256 envanterini çıkar.
4. Deterministic payload-composite setter ve simulator readback smoke testini yaz.
5. S0a/S0b runner’larını 384 environment ile çalıştır.
6. S1 static payload grid’ini çalıştır; ham artefact ve eligibility durumlarını üret.
7. S2 deterministic switch runner’ını çalıştır; SysID `P_hat(t)` trace’ini kaydet.
8. S3’ü MLP kalibrasyonundan sonra çalıştır.
9. Scorecard ve diagnostic atlas’ı aynı artefact’lerden üret.

## 13. Implementation öncesi sabitlenecek açık parametreler

Bu parametreler kod yazılmadan önce, sonuçlara bakılmadan sabitlenir:

- Payload-composite mapping: `added_mass(s) → com_x(s)`; işaret, ölçek ve bütün grid noktaları.
- Oracle headroom için asgari **mutlak** tracking-error marjı.
- S2’de switch öncesi ve switch sonrası ölçüm penceresi uzunluğu.
- S1/S2 `near_ood` ve `far_ood` grid noktalarının nihai listesi.
- S3 kick/terrain kalibrasyon taramasının aday aralıkları.
- Oracle seed seçimi: best-of-seeds’in hangi hücre/severity seviyesinde yapılacağı ve aynı seçimin bütün methodlar için nasıl sabit tutulacağı.

Bu alanlar kapandıktan sonra suite hücreleri ve scoring kuralları sonuçlara göre değiştirilmeyecektir.
