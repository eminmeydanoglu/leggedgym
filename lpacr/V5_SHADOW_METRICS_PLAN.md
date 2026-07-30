# V5 Uniform Shadow Metrics Deney Planı

## 1. Amaç

V5'in mevcut stage-return learning-progress (LP) sinyali, hücrelerin gelecekteki
held-out gelişimini güvenilir biçimde öngörmüyor. Bu deneyin amacı yeni bir
sampler'ı doğrudan devreye almak değil; **tek bir Uniform V5 eğitiminde** aşağıdaki
aday sinyaller için gerekli ham verileri birlikte toplamak ve daha sonra aynı
checkpoint/validation protokolünde karşılaştırmaktır:

1. Positive Value Loss (PVL):
   `mean(max(raw_gae, 0))`
2. Mutlak avantaj:
   `mean(abs(raw_gae))`
3. Yumuşatılmış başarı ilerlemesi:
   stage'ler boyunca `success_rate` trendi
4. Frontier:
   `4 * p * (1 - p)`, burada `p` hücrenin başarı oranıdır

Birincil araştırma sorusu:

> Stage `t` sırasında ölçülen hücre sinyali, aynı hücrenin sabit held-out
> validation bankındaki `t+2` ve `t+4` stage gelişimini öngörüyor mu?

Bu çalışma bir sampler karşılaştırması değildir. Eğitim boyunca görev dağılımı
Uniform kalır ve aday sinyaller yalnızca shadow telemetry olarak kaydedilir.

## 2. Sabitlenen deney sınırları

- Ortam, robot, reward, observation, PPO, domain randomization, terrain grid,
  task codec, standstill oranı ve toplam eğitim bütçesi mevcut V5 Uniform
  kontratından değişmez.
- Eğitim kolu yalnızca `uniform` olur.
- PVL, `|GAE|`, success progress veya frontier eğitim sırasında sampling
  olasılıklarını değiştirmez.
- `rolling_completion` kullanılmaz. V5 stage saati korunur.
- Standstill episode/timestep'leri hiçbir moving-cell metriğine katılmaz; ayrı
  reserved-bucket telemetry olarak kalır.
- Held-out validation eğitim sürecinin içine sokulmaz. Belirlenen stage
  checkpoint'leri eğitim bittikten sonra aynı sabit bank üzerinde değerlendirilir.
- Mevcut `best_spnte.pt` seçim kontratı değiştirilmez. Bu deneyin ara
  checkpoint'leri model seçmek için değil, ileri-gelişim hedefi üretmek içindir.

## 3. Ön koşul: stage sansürünü kaldırma

### 3.1 Mevcut problem

Mevcut stage estimator yalnızca
`assigned_revision == current_sampler_revision` olan tamamlanmaları stage
ortalamasına alıyor. Stage sınırını geçen episode'lar gerçek ve tamamlanmış
olmalarına rağmen LP istatistiğinden atılıyor. Kayıp yaklaşık `%46` ve performans
kör değil: uzun, çoğunlukla daha başarılı episode'lar sınırı daha sık geçiyor.

### 3.2 Yeni stage üyelik kuralı

Stage istatistiğinin üyelik kuralı:

> Açık stage'in zaman aralığında tamamlanan bütün gerçek moving episode'lar,
> hangi sampler revision altında atanmış olursa olsun, tamamlandıkları stage'in
> hücre istatistiğine girer.

`assigned_revision` silinmez; provenance ve gecikme analizi için kaydedilmeye
devam eder. Ancak stage metriğine kabul kapısı olmaz. Böylece stage saati ve
sampler revision mekanizması korunurken veri sansürü kalkar.

### 3.3 Korunacak lifecycle kuralları

- Outcome, reset sırasında yeni görev atanmadan önce eski `active_task_id` ile
  toplanır.
- Startup'ta hiç step almamış ghost episode'lar kabul edilmez.
- Standstill outcome'ları moving-cell accumulator'larına girmez.
- Her completion tam olarak bir kez sayılır.
- Stage kapandıktan sonra gelen completion geriye dönük eski snapshot'ı
  değiştirmez; tamamlandığı açık stage'e yazılır.
- Resume sonrasında kaydedilmiş completion veya accumulator tekrar sayılmaz.
- `transition_occupancy` içindeki
  `assigned_revision:completion_revision` provenance korunur.

### 3.4 Yeni tanı alanları

Her stage snapshot'ında en az şu alanlar bulunmalıdır:

- `completion_stage_episode_count[cell]`
- `assigned_same_revision_count[cell]`
- `cross_revision_completion_count[cell]`
- `cross_revision_completion_fraction[cell]`
- `completion_age_revisions` özeti
- mevcut cumulative `task_assignment_count[cell]`
- mevcut cumulative `task_completion_count[cell]`

Eski kayıtlarla yeni kayıtlar karıştırılmamalıdır. Telemetry/checkpoint schema
version artırılmalı ve stage-admission semantiği config fingerprint/provenance
içinde açıkça yer almalıdır.

## 4. Shadow telemetry tasarımı

### 4.1 Timestep–hücre eşlemesi

Rollout storage'a her transition için aşağıdakiler eklenir:

- action alınırken aktif olan `task_id`
- `is_standstill`
- gerekirse `sampler_revision`

Kimlik **`env.step()` öncesinde** yakalanmalıdır. Done/reset sonrasında
adapter'daki yeni görev kimliğini okuyup geçmiş transition'a yazmak, terminal
transition'ı yanlış hücreye bağlar.

Beklenen şekiller:

```text
task_ids:     [num_steps_per_env, num_envs]
standstill:   [num_steps_per_env, num_envs]
```

Bu alanlar PPO minibatch girdisi değildir; yalnızca stage shadow aggregation
için kullanılır.

### 4.2 Raw GAE sınırı

Mevcut rollout storage önce

```text
raw_gae = returns - values
```

hesaplayıp daha sonra bütün rollout üzerinde normalize ediyor. Shadow metrikler
**normalizasyondan önceki** `raw_gae` üzerinden hesaplanmalıdır.

Normalize edilmiş advantage kullanılmayacaktır; çünkü değeri aynı rollout'taki
diğer hücrelerin ortalama ve standart sapmasına bağlıdır ve stage'ler arası
karşılaştırılabilir bir büyüklük değildir.

PPO'nun kullandığı normalize edilmiş advantage ve öğrenme davranışı byte-level
olarak mümkün olduğunca korunmalıdır. Shadow hesaplama detached/read-only
olmalı; autograd grafiği veya optimizer girdisi oluşturmamalıdır.

### 4.3 Hücre başına kaydedilecek yeterli istatistikler

PVL ve `|GAE|` için yalnızca son ortalamayı kaydetmek yerine stage başına:

- `gae_timestep_count`
- `raw_gae_sum`
- `raw_gae_sq_sum`
- `positive_gae_sum`
- `positive_gae_sq_sum`
- `absolute_gae_sum`
- `absolute_gae_sq_sum`
- tercihen sabit quantile sketch veya sınırlı histogram

kaydedilir. Böylece ortalama, varyans, SEM, pozitif oran ve uç değer duyarlılığı
post-hoc yeniden hesaplanabilir.

Türetilmiş shadow değerler:

```text
pvl = positive_gae_sum / gae_timestep_count
abs_gae = absolute_gae_sum / gae_timestep_count
positive_gae_fraction = count(raw_gae > 0) / gae_timestep_count
```

Bir rollout stage sınırını geçiyorsa timestep'ler rollout'un bittiği stage'e
yığılmamalıdır. Aggregator, transition'ın gerçekleştiği global control-step
aralığına göre stage'e yazmalı veya stage sınırında accumulator'ı doğru biçimde
bölmelidir.

### 4.4 Başarı ve frontier için ham istatistikler

İlk online başarı tanımı:

```text
survival_success = terminal_reason == "timeout"
```

Her completion için ayrıca episode length ve terminal reason korunur. Hücre/stage
başına:

- `success_count`
- `completion_count`
- `terminal_count`
- `timeout_count`
- episode-length sum/square-sum

kaydedilir.

Ham `success_rate = success_count / completion_count` Atlas'a yazılır. Aşağıdaki
dönüşümler eğitim kodunda sampler skoru olarak sabitlenmez; analizde birden çok
ufuk/parametreyle türetilir:

- 4-stage ve 8-stage EWMA
- ağırlıklı doğrusal trend
- Beta-Binomial shrinkage ile success trendi
- `frontier = 4p(1-p)`
- belirsizlik-cezalı frontier varyantı

Survival success yalnız erişilebilirlik proxy'sidir. Nihai öğrenme hedefi olarak
yorumlanmaz; robot timeout'a ulaşırken kötü tracking yapabilir.

### 4.5 Log ve artefact sözleşmesi

- Stage snapshot tek atomik commit olarak yayımlanır.
- NDJSON/SSE yolu mevcut dashboard gibi fail-open kalır; telemetry arızası
  eğitimi durduramaz.
- JSON'da gözlenmemiş değerler `null` olur; sahte `0` yazılmaz.
- TensorBoard'a yalnız küçük global özetler yazılır. 84 hücrelik ham diziler
  Atlas/NDJSON artefact'ında tutulur.
- Checkpoint, açık stage accumulator'larını ve shadow accumulator'larını
  continuation için saklar.
- Her run manifest'i şu provenance'ı içerir:
  - Git commit ve dirty-state
  - config fingerprint
  - telemetry/checkpoint schema version
  - stage-admission semantiği
  - seed
  - task-space/bank fingerprint
  - checkpoint iteration ↔ stage index eşlemesi

## 5. Checkpoint ve held-out validation protokolü

### 5.1 Ara checkpoint'ler

Stage sınırına en yakın güvenli PPO iteration'ında checkpoint alınır. En az:

```text
stage 0, 2, 4, 6, ... ve final
```

Checkpoint manifest'inde gerçek `stage_index`, `global_control_steps` ve PPO
iteration yazılır. Sadece dosya adındaki tahmini stage numarasına güvenilmez.

Checkpoint yazımı eğitimin RNG akışını veya sampler RNG'sini tüketmemelidir.

### 5.2 Sabit validation bank

Bütün checkpoint'ler aynı:

- task listesi ve C-order task kimliği,
- terrain/world seed'leri,
- episode/replica seed'leri,
- episode uzunluğu,
- policy runner/config,
- command desteği

ile değerlendirilir. Bank fingerprint'i her sonuç satırına yazılır.

Birincil held-out hücre metriği mevcut V5 doğrulama kontratındaki tracking ve
survival ölçümlerinden üretilir:

- `spnte_lin` değişimi
- survival/fall değişimi
- önceden tanımlı strict `cell_success`

Tracking ve survival ayrı raporlanır; tek bir başarı yüzdesine eritilip mekanizma
kaybedilmez.

### 5.3 İleri-gelişim hedefleri

Her hücre ve ölçülebilir stage için:

```text
gain_spnte_i(t, h) =
    spnte_i(t) - spnte_i(t+h)      # pozitif = iyileşme

gain_success_i(t, h) =
    success_i(t+h) - success_i(t)

h ∈ {2, 4} stage
```

Stage `t` shadow sinyalleri yalnız gelecekteki checkpoint hedefleriyle
eşleştirilir. Aynı pencerenin training return'ü “gelecek kazanç” hedefi olarak
kullanılmaz.

## 6. Analiz planı ve karar kapıları

### 6.1 Birincil analiz

Her aday sinyal için:

- hücreler arası Spearman korelasyonu:
  `signal_i(t) ↔ future_gain_i(t,h)`
- stage-clustered bootstrap güven aralığı
- top-k sinyal hücrelerinin sonraki held-out kazancı
- top-k lift'in aynı büyüklükte rastgele hücre seçimine karşı farkı
- erken/geç eğitim rejimi ayrımı
- terrain family, level ve `v_x` bin tabakaları
- sinyal örnek sayısı/SEM'e göre duyarlılık

Birincil ufuklar `h=2` ve `h=4`; bunlardan biri sonuç görüldükten sonra seçilip
tek başarı ufku ilan edilmez.

### 6.2 Null ve negatif kontroller

- Hücre kimliklerini stage içinde permüte ederek null korelasyon/lift dağılımı.
- Sinyali geleceğe değil geçmiş kazanca kaydırarak yön/kaçak kontrolü.
- Uniform sampling probability'nin gelecek kazancı öngörmemesi.
- Normalize GAE ile raw GAE sonuçlarını yalnız teşhis amaçlı karşılaştırma;
  normalize sonuç birincil aday olamaz.
- Episode count, mean episode length ve ham return'ü nuisance/baseline predictor
  olarak raporlama.
- Aynı stage verisinden türeyen predictor/target kullanımını engelleyen indeks
  testi.

### 6.3 PVL için özel kontroller

PVL yüksekliği critic'in sistematik düşük tahmininden gelebilir. Bu nedenle:

- mean raw GAE ve positive fraction
- value loss / return–value residual
- hücre başına örnek sayısı
- stage ve checkpoint yaşı

ile ilişkisi raporlanır. PVL ancak held-out ilerlemeyi, yalnız critic hatasını
veya episode uzunluğunu açıklamanın ötesinde öngörüyorsa aday sampler sinyalidir.

### 6.4 Başarı ölçütü

Bir sinyal sonraki sampler pilotuna ancak:

1. yönü önceden tanımlı held-out gain ile tutarlıysa,
2. bootstrap CI yalnız permütasyon null'ını tekrarlamıyorsa,
3. top-k lift rastgele seçime göre pozitif ve pratik olarak anlamlıysa,
4. yalnız tek terrain/velocity diliminden kaynaklanmıyorsa,
5. erken veya geç rejim iddiası açıkça sınırlandırılmışsa,
6. ölçüm sayısı/episode length nuisance'ının basit vekili değilse

taşınır.

Tek seed gözlemsel koşu, yöntem üstünlüğü veya nihai sampler zaferi ilan etmek
için yeterli değildir. Bu koşu **sinyal eleme ve enstrümantasyon doğrulama**
deneyidir.

## 7. Test planı

### 7.1 Saf birim testleri — yerel CPU

Stage admission:

- aynı revision completion kabul edilir;
- eski revision completion açık completion-stage'e kabul edilir;
- gelecekteki revision reddedilir;
- late completion tam bir kez sayılır;
- standstill moving-cell sayacına girmez;
- ghost episode girmez;
- hücreler arası count/sum/square-sum doğru ayrılır.

GAE aggregation:

- elle hesaplanmış küçük tensor için raw GAE, PVL ve `|GAE|` eşleşir;
- global advantage normalization shadow değerleri değiştirmez;
- positive/zero/negative GAE sınırları doğrudur;
- standstill maskesi çalışır;
- terminal transition reset sonrası yeni task ID'ye yazılmaz;
- rollout içindeki task değişimi iki hücreye doğru bölünür;
- stage sınırını geçen rollout doğru stage parçalarına ayrılır.

Success/frontier:

- timeout success, terminal failure olarak sayılır;
- `p=0` ve `p=1` frontier `0`, `p=0.5` frontier `1` olur;
- gözlenmemiş hücre `null`, gerçek sıfırdan ayrılır;
- EWMA/trend analizi sentetik artan, düz ve azalan serilerde doğru yönü verir.

Numerik güvenlik:

- boş hücre;
- tek örnek;
- çok büyük sonlu GAE;
- `NaN/Inf` girdisinin açıkça reddedilmesi;
- sum/square-sum'dan negatif yuvarlama varyansının clamp edilmesi.

### 7.2 Lifecycle ve checkpoint entegrasyon testleri — yerel CPU

- `action task_id → env.step → done → old outcome → reset → new assignment`
  sırasını gerçek adapter/runner sınırında doğrulama.
- Stage kapanırken aynı reset batch'indeki outcome'ların hangi stage'e
  yazıldığını sabitleyen regression testi.
- Save/load sonrası açık stage'in bütün return, GAE, success ve provenance
  accumulator'larının eşitliği.
- Kesintisiz koşu ile checkpoint/resume koşusunun bir sonraki snapshot'ının
  eşitliği.
- Eski schema checkpoint'inin ya kontrollü migrate edilmesi ya da anlaşılır,
  erken hata vermesi.
- Telemetry listener kapalı/açık/hata atarken eğitim lifecycle'ının aynı
  sonucu üretmesi.

### 7.3 Property ve sentetik istatistik testleri — yerel CPU

- Rastgele completion dizilerinde:
  `sum(per-cell counts) == admitted moving completions`.
- Assignment/completion revision farkı admission toplamını değiştirmez.
- Her timestep ya tam bir moving hücreye ya standstill bucket'a aittir; ikisine
  birden veya hiçbirine ait olamaz.
- Sentetik learnability sinyali enjekte edildiğinde ileri-tahmin analizi doğru
  metriği null metrikten ayırır.
- Tamamen permüte/null veride başarı kapısının yanlış pozitif üretme oranı
  bootstrap/permutation toleransıyla uyumludur.

### 7.4 Dashboard/artefact sözleşme testleri — yerel

- Schema alanlarının NDJSON serialize/parse round-trip'i.
- `null`/finite sayı sözleşmesi.
- Eski Atlas run'larının yeni loader ile okunabilmesi.
- Dashboard alan eksikken fail-open davranış.
- `npm test` ile dashboard regresyonları.
- Üretilen küçük fixture üzerinde analiz raporunun deterministik hash veya
  golden-summary kontrolü.

### 7.5 Genesis smoke test — UHeM compute node

Yerel unit/integration testleri geçtikten sonra izole kaynak kopyası UHeM'e
aktarılır. Login node yalnız kısa `rsync`, Slurm ve log/status işlemleri için
kullanılır; Genesis çalıştırma `ssh makine` ile ulaşılan compute node'da yapılır.

Smoke koşusu:

- gerçek V5 Uniform task registration;
- gerçek Genesis env;
- küçük env sayısı;
- en az bir PPO update;
- zorunlu olarak en az bir episode reset/completion;
- mümkünse yapay kısaltılmış stage ile en az iki stage sınırı;
- checkpoint save/load;
- Atlas frame üretimi.

Smoke kabul kanıtı:

- `hostname` compute node'dur ve GPU görünür;
- task ID shape/range doğrudur;
- raw GAE istatistikleri sonlu ve non-empty'dir;
- moving timestep sayısı ile hücre toplamı tutarlıdır;
- cross-revision completion gerçekten kabul edilmiştir;
- stage snapshot/NDJSON/checkpoint alanları doludur;
- eğitim telemetry kapalıyken de devam eder;
- loss/reward akışında `NaN/Inf` yoktur.

Smoke yalnız “process exit 0” ile başarılı sayılmaz; üretilen checkpoint ve
Atlas artefact'ı indirip içerik doğrulaması yapılır.

### 7.6 Orta ölçekli pilot — UHeM compute node

Tam koşudan önce yaklaşık 5–8 stage kapsayan Uniform pilot:

- production `num_envs` mümkünse korunur;
- gerçek stage uzunluğu kullanılır;
- en az bir ara checkpoint ve resume sınırı içerir;
- dashboard/telemetry fail-open yolu gerçek eğitimle sınanır;
- stage sansürünün `%0` olduğu completion sayaçlarından doğrulanır;
- eski revision completion oranı ve revision-age dağılımı raporlanır;
- shadow metriklerin 84 hücrede coverage/SEM dağılımı incelenir;
- logging'in iteration wall-time ve GPU memory overhead'i ölçülür.

Kabul kapıları:

- completion muhasebesi tamdır;
- checkpoint/resume devamlılığı geçer;
- telemetry overhead'i önceden belirlenen makul sınırdadır;
- stage başına çoğu hücre için analiz yapılabilir GAE ve completion örneği vardır;
- artefact büyümesi tam koşuda disk/quota riski yaratmaz.

### 7.7 Tam Uniform shadow eğitimi — UHeM

Yalnız bütün önceki kapılar geçince:

- temiz, fingerprint'li kaynak ağacı;
- önceden kaydedilmiş seed ve run adı;
- tam V5 bütçesi;
- stage checkpoint manifest'i;
- append-only Atlas/telemetry;
- Slurm logları ve periyodik sağlık kontrolleri

ile çalıştırılır.

Eğitim sonunda otomatik artefact envanteri çıkarılır. Eksik stage, checkpoint,
manifest veya telemetry varsa validation başlamadan önce açıkça raporlanır.

### 7.8 Held-out validation — UHeM

Ara checkpoint'ler aynı sabit bankta, tercihen bağımsız ve yeniden başlatılabilir
Slurm array/batch işleriyle değerlendirilir. Her iş:

- checkpoint SHA256;
- bank fingerprint;
- config/runner fingerprint;
- seed aralığı;
- tamamlanan/planlanan episode sayısı;
- hata/eksik hücre listesi

üretir.

Eksik veya farklı fingerprint'li checkpoint sonucu aynı ileri-gelişim eğrisine
katılmaz. Validation sonucu eğitim dashboard izlenimiyle ikame edilmez.

## 8. Uygulama sırası

1. Stage completion-admission semantiğini değiştir ve schema'yı artır.
2. Stage sansürü/lifecycle/checkpoint testlerini tamamla.
3. Rollout storage'a timestep task provenance'ı ekle.
4. Raw GAE shadow aggregation ve yeterli istatistikleri ekle.
5. Success/frontier için completion istatistiklerini ekle.
6. Checkpoint, Atlas, loader ve dashboard sözleşmesini genişlet.
7. Yerel unit/integration/property/dashboard testlerini çalıştır.
8. UHeM compute node'da gerçek Genesis smoke yap.
9. UHeM'de 5–8 stage pilot ve overhead/coverage incelemesi yap.
10. Kabul kapıları geçerse tam Uniform shadow eğitimini başlat.
11. Stage checkpoint'lerini sabit bankta UHeM'de değerlendir.
12. Önceden tanımlı `h={2,4}` analizini üret ve yalnız doğrulanan sinyali
    sonraki sampler pilotuna aday göster.

## 9. Bu planın üretmeyeceği sonuçlar

Bu tek Uniform koşu sonunda aşağıdakiler söylenmeyecektir:

- “PVL, frontier veya başka bir yöntem Uniform'dan daha iyi eğitir.”
- “Tek seed'de en yüksek korelasyon gösteren metrik kazanmıştır.”
- “Survival frontier doğrudan tracking öğrenilebilirliğidir.”
- “Training return artışı held-out gelişim kanıtıdır.”
- “Bir dashboard görüntüsü telemetry/lifecycle doğrulamasıdır.”

Bu deneyin geçerli nihai çıktısı şudur:

> Aynı Uniform eğitim akışında, sansürsüz ve provenance'ı korunmuş veriden
> ölçülen aday sinyallerin hangilerinin gelecekteki sabit-bank hücre gelişimiyle
> güvenilir ilişki gösterdiği ve hangilerinin sampler pilotuna taşınmaya değer
> olmadığı.
