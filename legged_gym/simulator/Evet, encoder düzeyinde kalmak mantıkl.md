Evet, encoder düzeyinde kalmak mantıklı. Hatta bence RoboGauge’ın temel fikri çöpe atılacak bir fikir değil. **Dense privileged teacher + proprioceptive MoE student** oldukça iyi bir araştırma problemi.

Ama mevcut uygulamada yanlış olan şey şu:

> Tek bir teacher olması değil, bütün uzmanların yalnızca karıştırıldıktan sonra tek teacher latent’ine eşlenmesi.

Bu ayrım çok önemli.

## Tek dense teacher neden problem değil?

Teacher’ın tek MLP olması uzmanlaşmayı teorik olarak engellemez. Teacher bütün durum uzayını temsil eden tek bir fonksiyon üretsin:

[
z_T = f_T(s_{\text{priv}})
]

Student MoE bu fonksiyonun farklı bölgelerini farklı uzmanlarla yaklaşık olarak öğrenebilir:

[
z_k = E_k(h), \qquad
z_S = \sum_k g_k(h)z_k.
]

Örneğin:

* Bir expert düz zemin ve düşük temas belirsizliği bölgesini,
* Bir expert merdiven çıkışını,
* Bir expert iniş ve negatif yük değişimini,
* Bir expert kayganlık, darbe veya yüksek model hatası rejimini

öğrenebilir.

Burada teacher’ın kendisinin MoE olması gerekmiyor. Tek global fonksiyon, parça parça yerel fonksiyonlarla yaklaşık olarak temsil edilebilir.

Fakat RoboGauge expert’lara **hangi parçanın kendilerine ait olduğunu söylemiyor.**

## Mevcut loss neden uzmanlaştırmıyor?

Mevcut yapı:

[
z_S=\sum_k g_kz_k,
\qquad
L_{\text{latent}}=|z_S-z_T|^2.
]

Bir expert çıktısına gelen gradient:

[
\frac{\partial L}{\partial z_k}
===============================

2g_k(z_S-z_T).
]

Gate uniformsa:

[
g_1=g_2=\dots=g_K=\frac1K,
]

bütün expert’lar aynı residual yönünde, aynı büyüklükte güncelleniyor:

[
\frac{\partial L}{\partial z_1}
===============================

# \frac{\partial L}{\partial z_2}

# \dots

\frac{\partial L}{\partial z_K}.
]

Dolayısıyla senin söylediğin sezgi esasen doğru: **ortak loss, uniform gate altında expert’ları aynı teacher hedefinin aynı yönüne taşıyor.**

Router açısından da bir tavuk-yumurta problemi var. Router’ın gradient’i expert çıktılarının birbirinden farklı olmasına bağlı. Expert’lar birbirine benziyorsa gate’i değiştirmenin mixed latent üzerinde etkisi kalmıyor. Bu durumda latent loss router’a anlamlı sinyal vermiyor; geriye kalan load-balance loss ise doğrudan uniform kullanım çözümünü destekliyor.

RoboGauge kodunda student encoder yalnız mixed latent üzerinden teacher’a eşleniyor; PPO actor yolunda student encoder `no_grad` altında, critic tarafında da latent detach ediliyor. Bu nedenle expert’lar davranış başarısından doğrudan farklılaşan bir sinyal almıyor. Ayrıca mevcut “expert” yapısı büyük ölçüde ortak trunk ve küçük ayrı head’lerden oluşuyor. ([GitHub][1])

## Daha derin problem: Student teacher’ın gördüğü bilgiyi görmüyor

Bence burası uniform gate probleminden bile daha önemli.

Teacher:

[
z_T=f_T(s_{\text{priv}})
]

hesaplarken terrain height, temas kuvvetleri, dinamik parametreler gibi privileged bilgi görüyor. Student ise yalnız kısa proprioceptive history görüyor:

[
h=(o_{t-H+1},\dots,o_t).
]

Aynı history’nin birden fazla privileged duruma karşılık gelmesi mümkün. Mesela robot henüz basamağa temas etmeden:

* Önünde düz zemin olabilir,
* Bir basamak olabilir,
* Bir boşluk olabilir.

Student’ın history’si aynı veya çok benzerken teacher latent’leri farklı olabilir.

Deterministik L2 regresyonun optimumu:

[
f_S^*(h)
========

\mathbb E[z_T\mid h].
]

Yani student, ayırt edemediği teacher latent’lerinin **koşullu ortalamasını** öğrenir.

MoE kullanmak bunu mevcut formda çözmüyor. Çünkü loss expert’lara ayrı ayrı değil, ortalamalarına uygulanıyor:

[
L
=

\left|
\sum_k g_k z_k-z_T
\right|^2.
]

Bu hâlâ tek bir deterministik tahmin. Birden fazla hipotezi korumak yerine, hipotezleri önce karıştırıp sonra L2 uyguluyorsunuz. Sonuç yine ortalamaya çöküyor.

Orijinal CTS çalışması da privileged ve student gözlemleri arasındaki fark nedeniyle tam latent taklidinin zor olabileceğini ve yalnız teacher taklidinin student için her zaman en iyi amaç olmayabileceğini belirtiyor. 

Bu yüzden yapılması gereken temel değişiklik şu:

> **Loss of mixture yerine mixture of losses kullanmak.**

## RoboGauge’ı kurtaracak temel değişiklik

Mevcut:

[
L_{\text{old}}
==============

d\left(
\sum_k g_kz_k,,
z_T
\right).
]

Olması gereken yapı:

[
L_{\text{local}}
================

\sum_k q_k,d(z_k,z_T).
]

Burada (q_k), o örnek için hangi expert’ın sorumlu olduğunu belirleyen assignment veya responsibility.

Bu küçük görünen değişiklik, expert’ların gradient’ini tamamen değiştiriyor:

[
\frac{\partial L_{\text{local}}}{\partial z_k}
==============================================

q_k\frac{\partial d(z_k,z_T)}{\partial z_k}.
]

Bir sample için:

```text
q = [0.92, 0.04, 0.03, 0.01]
```

ise birinci expert güçlü biçimde öğrenirken diğerleri çok az veya hiç öğrenmiyor. Expert’lar artık aynı batch içindeki bütün örnekleri aynı şekilde takip etmiyor.

Bunu uygulamanın iki temel yolu var.

### 1. Winner-take-all / balanced assignment

Her sample için her expert’ın teacher’a olan hatasını hesapla:

[
d_{nk}=d(z_{nk},z_{T,n}).
]

Sonra en iyi expert’ı seç:

[
k_n^*=\arg\min_k d_{nk}.
]

Expert loss:

[
L_{\text{expert}}
=================

d(z_{n,k_n^*},z_{T,n}).
]

Router loss:

[
L_{\text{router}}
=================

\operatorname{CE}(g(h_n),k_n^*).
]

Fakat doğrudan argmin kullanırsan başlangıçta şans eseri iyi olan bir expert bütün örnekleri ele geçirebilir. Bu nedenle assignment’ların batch boyunca dengelenmesi gerekir. Sinkhorn assignment, kapasite sınırı veya terrain-stratified batch kullanılabilir.

### 2. Probabilistic mixture distillation

Daha yumuşak ve teorik olarak daha doğru versiyon:

[
L_{\text{NLL}}
==============

-\log
\sum_k
g_k(h)
\exp\left(
-\frac{d(z_k,z_T)}{2\sigma^2}
\right).
]

Bundan doğal responsibility çıkar:

[
q_k
===

\frac{
g_k\exp(-d_k/2\sigma^2)
}{
\sum_j g_j\exp(-d_j/2\sigma^2)
}.
]

Teacher latent’e en yakın expert daha fazla gradient alır. Bu gerçek bir mixture modeli davranışıdır.

Ancak yalnız bunu eklemek de başlangıç simetrisini her zaman kıramaz. Expert’lar başlangıçta aynıysa responsibility’ler de aynı olabilir. Bu nedenle privileged routing supervision veya balanced assignment ile birlikte kullanılmalı.

## Benim önerim: Privileged-Routed Residual MoE-CTS

RoboGauge’ın teacher’ını dense bırakırdım. Teacher’a MoE eklemezdim.

Bunun yerine training sırasında kullanılan hafif bir **teacher routing head** eklerdim:

[
q_T
===

R_T(c_{\text{priv}}).
]

Buradaki (c_{\text{priv}}):

* Terrain height örnekleri,
* Eğim ve roughness istatistikleri,
* Temas kuvvetleri,
* Sürtünme,
* Payload ve motor strength,
* Command,
* Gerekirse temas veya gait phase

gibi training sırasında bilinen bilgileri içerir.

Student router:

[
g_S=R_S(h)
]

ise yalnız deploy edilebilir history’den tahmin yapar.

Router distillation:

[
L_{\text{route}}
================

D_{\mathrm{KL}}
\left(
\operatorname{stopgrad}(q_T)
;|;
g_S
\right).
]

Bu yaklaşım başka MoE alanlarında kullanılan teacher-guided routing fikrine çok yakın: dense teacher’ın ara özelliklerinden training-only routing hedefleri üretiliyor ve student router bu hedefleri distill ediyor. Bu, router’ı yalnız task loss’un zayıf ve dolaylı gradient’ine bırakmıyor. ([arXiv][2])

Buradaki önemli fark şu:

> Teacher expert üretmiyor. Teacher yalnız student expert’ları için bir partition, yani iş bölümü üretiyor.

## Terrain label kullanmalı mıyız?

Kullanılabilir ama yalnız terrain class’ı kullanmak biraz kaba olur.

Örneğin “stairs” tek bir kontrol rejimi değil:

* Merdivene yaklaşma,
* Ön bacak temas anı,
* Gövdeyi yükseltme,
* Arka bacakların basamağa çıkması,
* Merdivenden ayrılma

farklı latent ihtiyaçları oluşturabilir.

Ayrıca aynı rough terrain:

* Yavaş hızda,
* Yüksek hızda,
* Farklı payload altında

başka bir kontrol rejimine dönüşebilir.

Bu nedenle router hedefini yalnız:

```text
flat / stairs / slope / obstacle
```

olarak tanımlamak yerine **privileged control context** üzerinde tanımlamak daha iyi.

İki seçenek var:

1. Terrain ve dynamics metadata’sını bir context encoder’a verip doğrudan supervised routing yapmak.
2. Privileged context veya teacher hidden feature’larını dengeli biçimde K prototype’a cluster etmek.

İkinci yaklaşımda:

[
q_{nk}
\propto
\exp\left(
\frac{\operatorname{sim}(c_n,p_k)}{\tau}
\right),
]

burada (p_k) öğrenilen context prototype’larıdır. Sinkhorn ile batch assignment’ları dengelenebilir.

Böylece expert’lar insanların verdiği nominal terrain isimlerini değil, teacher’ın kontrol açısından önemli bulduğu rejimleri paylaşır.

Mutual-information tabanlı MoE çalışmalarının ana sezgisi de bu: yalnız marginal expert kullanımını dengelemek yerine, **input/context ile expert seçimi arasındaki bağı güçlendirmek** gerekir. Yüksek batch entropy ve düşük sample entropy birlikte kullanılmalıdır. ([arXiv][3])

Matematiksel olarak:

[
I(C;K)
======

H(K)-H(K\mid C).
]

Bunu maksimize etmek yaklaşık olarak:

* Batch boyunca expert kullanımını çeşitli tutmak,
* Tek bir sample için gate’i keskin tutmak,
* Benzer context’leri aynı expert’a göndermek

demektir.

RoboGauge yalnız ilk maddeyi yapıyor.

## Student mimarisini de değiştirmek gerekiyor

Mevcut büyük shared trunk + küçük expert head tasarımı uzmanlaşmayı zayıflatıyor. Çünkü temsilin büyük kısmı expert ayrımından önce oluşturuluyor.

Ama tamamen bağımsız sekiz büyük MLP de gereksiz ve kararsız olabilir. En temiz ara çözüm **shared base + independent residual experts**:

[
u=\phi(h)
]

[
z_0=B(u)
]

[
r_k=E_k(u)
]

[
z_k=\operatorname{Norm}(z_0+\alpha r_k).
]

Burada:

* (\phi): temporal stem, örneğin TCN, GRU veya history MLP,
* (z_0): bütün terrain’lerde ortak locomotion bilgisi,
* (r_k): terrain/dynamics rejimine özel düzeltme,
* Her (E_k): gerçekten bağımsız küçük MLP.

Bu tasarımın sezgisi şu:

> Expert’ların bütün locomotion temsilini sıfırdan tekrar öğrenmesine gerek yok. Ortak proprioception ve command bilgisi shared base’de kalır; expert’lar teacher latent’inin rejime bağlı residual kısmını öğrenir.

Bu, expert’ların birbirine tamamen kopyalanmasını azaltırken ortak veri verimliliğini korur.

Başlangıçta dört expert ile başlardım. Yedi terrain var diye yedi veya sekiz expert gerekmiyor. Terrain sayısı ile kontrol rejimi sayısı aynı şey değil.

## Full expert latent’lerini orthogonal yapmak yanlış olur

Bazı MoE çalışmalarında expert output orthogonality veya diversity loss’u kullanılıyor. Bunun nedeni expert’ların aynı fonksiyona çökmesini önlemek. Ayrıca uniform routing’i engellemek için router-score variance teşvik ediliyor. ([arXiv][4])

Fakat RoboGauge’a doğrudan:

[
z_i^\top z_j\approx0
]

loss’u koymak doğru değil.

Çünkü bütün expert’ların gerektiğinde aynı teacher latent’in bazı bölgelerini yaklaşık olarak öğrenmesi gerekiyor. Hem teacher’a yakın olmalarını hem de birbirlerine orthogonal olmalarını istemek çelişebilir.

Diversity uygulanacaksa full latent yerine residual’lara uygulanmalı:

[
L_{\text{res-div}}
==================

\sum_{i\neq j}
\left|
\frac{r_i^\top r_j}
{|r_i||r_j|}
\right|.
]

Daha da iyisi, bunu bütün sample’larda değil yalnız aynı sample için aktif olmayan veya assignment sınırında bulunan expert’lara küçük katsayıyla uygulamak.

Ben bunu ilk aşamada zorunlu loss olarak görmüyorum. **Doğru assignment ve mixture-of-losses**, diversity regularizer’dan daha önemli.

## Latent L2 tek başına yeterli değil

Teacher latent’in koordinatlarının hepsi action açısından eşit önemli değil.

Teacher encoder ve actor birlikte eğitildiği için teacher latent:

* Kontrol için kritik bilgiler,
* Actor’ın neredeyse kullanmadığı bilgiler,
* Birbirinin yerine geçebilen temsil yönleri

içerebilir.

Student’ın her latent boyutunu eşit L2 ile taklit etmesi gereksiz olabilir.

Bu nedenle per-expert mesafeyi yalnız latent hatasıyla değil, **action-aware distillation** ile tanımlardım:

[
\mu_T
=====

\pi(o,z_T),
]

[
\mu_k
=====

\pi(o,z_k).
]

[
d_k
===

\lambda_z
\left(1-\cos(z_k,z_T)\right)
+
\lambda_a
|\mu_k-\mu_T|^2.
]

İstenirse value farkı da eklenebilir:

[
+\lambda_v
|V(o,z_k)-V(o,z_T)|^2.
]

Burada actor veya critic MoE olmuyor. Aynı shared actor yalnız bir **ölçüm fonksiyonu** olarak kullanılıyor. Encoder expert’ının teacher’dan davranış açısından ne kadar saptığını ölçüyoruz.

Encoder update sırasında actor parametreleri freeze edilir; gradient yalnız student expert’a gider.

Bu sayede bir expert’ın teacher latent’ini sayısal olarak biraz farklı üretmesi ama aynı doğru action’ı çıkarması cezalandırılmaz. Tersine, küçük latent farkıyla ciddi action farkı yaratıyorsa güçlü biçimde cezalandırılır.

Farklı expert’ların sürekli birbirine veya aynı teacher hedeflerine aşırı distill edilmesi, expert çeşitliliğini azaltabilen bilinen bir problem. Bu yüzden her expert’ın her sample’ı taklit etmesi yerine yerel responsibility kullanmak daha mantıklı. ([arXiv][5])

## Önerdiğim toplam loss

Her batch sample’ı için:

[
z_T=\bar f_T(s_{\text{priv}})
]

Burada (\bar f_T), teacher encoder’ın EMA kopyası.

Student:

[
u=\phi(h),
\quad
z_k=\operatorname{Norm}(z_0+r_k),
\quad
g=R_S(u).
]

Privileged router:

[
q_T=R_T(c_{\text{priv}}).
]

Expert kontrol hatası:

[
d_k
===

\lambda_z d_{\text{latent}}(z_k,z_T)
+
\lambda_a d_{\text{action}}(\pi(o,z_k),\pi(o,z_T)).
]

Responsibility, privileged route ve expert hatasının birleşiminden üretilebilir:

[
q_k
\propto
q_{T,k}^{,\beta}
\exp\left(-\frac{d_k}{\tau}\right).
]

Ardından:

[
L_{\text{expert}}
=================

\sum_k
\operatorname{stopgrad}(q_k)d_k,
]

[
L_{\text{router}}
=================

D_{\mathrm{KL}}
\left(
\operatorname{stopgrad}(q)
|g
\right),
]

[
L_{\text{balance}}
==================

D_{\mathrm{KL}}
\left(
\frac1B\sum_n g_n
|p_{\text{target}}
\right),
]

[
L_{\text{sharp}}
================

\frac1B\sum_n H(g_n).
]

Toplam:

[
L=
L_{\text{expert}}
+\lambda_rL_{\text{router}}
+\lambda_bL_{\text{balance}}
+\lambda_hL_{\text{sharp}}
+\lambda_mL_{\text{mixed}}
+\lambda_tL_{\text{temporal}}.
]

Mixed latent imitation küçük bir yardımcı loss olarak kalabilir:

[
L_{\text{mixed}}
================

d\left(
\sum_k\tilde g_kz_k,,
z_T
\right),
]

ama ana expert eğitimi bu olmamalı.

(\tilde g), top-2 gate olabilir. İlk denemede top-1 yerine top-2 seçerdim; route hatalarının etkisi daha yumuşak olur.

## Teacher’ı neden EMA yapmak gerekir?

Mevcut sistemde teacher encoder da PPO ile sürekli değişiyor. Student hem hareketli teacher latent’ini hem de henüz oluşmamış expert partition’ını aynı anda takip etmeye çalışıyor.

Bu durum üç şeyi kararsızlaştırır:

* Expert assignment,
* Router hedefleri,
* Latent koordinat sistemi.

Teacher’ın tamamen dondurulması mümkün ama policy gelişirken eski kalabilir. Daha iyi seçenek:

[
\theta_{\bar T}
\leftarrow
\rho\theta_{\bar T}
+
(1-\rho)\theta_T.
]

Student distillation target’ı EMA teacher’dan gelir. PPO ise normal teacher encoder’ı güncellemeye devam eder.

Teacher-guided routing çalışmalarında teacher backbone’un stabil veya frozen olması routing supervision’ın güvenilirliği açısından önemli. RoboGauge uyarlamasında EMA aynı işlevi daha esnek biçimde görür. ([arXiv][2])

## Router’ın çözebileceği bilgi var mı?

Burada dürüst olmak gerekiyor:

Student router yalnız beş adımlık proprioceptive history görüyorsa, yaklaşan terrain’i temas öncesinde her zaman bilemez.

O durumda üç olasılık var:

1. Router temas sonrası dynamics rejimini öğrenir.
2. Router terrain yerine gait phase veya command’e uzmanlaşır.
3. Teacher route tahmin edilemediği için gate tekrar yüksek entropy’ye döner.

Bu bir optimizer sorunu değil, observability sorunu.

Bunu ölçmek için önce basit bir probe eğitmek gerekir:

[
h
\longrightarrow
\text{teacher route veya terrain/context class}.
]

Probe doğruluğu düşükse MoE router’ın başarılı olmasını beklemek anlamsızdır. Çözüm:

* History uzunluğunu artırmak,
* MLP yerine GRU veya temporal convolution kullanmak,
* Önceki action’ları ve contact bilgisini eklemek,
* Anticipatory terrain routing isteniyorsa depth veya height observation eklemek.

Exteroception eklenmeyecekse çalışmanın iddiasını da doğru koymak gerekir:

> “Terrain-semantic experts” yerine “proprioceptively identifiable dynamics-regime experts.”

Bu daha dürüst ve savunulabilir olur.

## Minimum kod değişikliğiyle kurtarma

RoboGauge kodunu mümkün olduğunca az değiştirmek için:

1. `MoEEncoder` mixed output yanında bütün `expert_outputs` değerlerini döndürsün.
2. Mixed latent MSE ana loss olmaktan çıkarılıp yardımcı loss olsun.
3. Her expert için teacher latent mesafesi hesaplansın.
4. Batch içinde balanced winner-take-all assignment yapılsın.
5. Yalnız atanmış expert teacher latent’i taklit etsin.
6. Gate bu assignment’ı cross-entropy ile tahmin etsin.
7. Inference ve training sırasında top-2 routing kullanılsın.
8. Teacher encoder’ın EMA kopyası distillation target’ı olsun.

Özet pseudocode:

```python
with torch.no_grad():
    teacher_latent = ema_teacher(privileged_obs)
    teacher_action = actor(obs, teacher_latent)

expert_latents, gate_logits = student_moe(history)
gate = gate_logits.softmax(dim=-1)

expert_actions = actor_for_each_latent(obs, expert_latents)

latent_error = cosine_distance(
    expert_latents,
    teacher_latent[:, None, :],
)

action_error = mse_per_expert(
    expert_actions,
    teacher_action[:, None, :],
)

cost = latent_error + action_coef * action_error

assignments = balanced_assignment(
    cost.detach(),
    privileged_context,
)

expert_loss = (assignments * cost).sum(dim=-1).mean()

router_loss = soft_cross_entropy(
    gate_logits,
    assignments.detach(),
)

student_loss = (
    expert_loss
    + router_coef * router_loss
    + small_balance_coef * balance_loss(gate)
)
```

Bu versiyon bile mevcut uniform çözümden çok daha iyi olur.

Ancak mevcut shared trunk + küçük heads nedeniyle expert farklılığı sınırlı kalabilir. Bu yüzden asıl önerim residual expert mimarisidir.

## Ben olsam hangi sırayla denerdim?

İlk deney:

* 4 expert,
* Shared temporal stem,
* Shared base latent,
* 4 bağımsız residual MLP,
* Top-2 gate,
* EMA dense teacher,
* Balanced per-expert assignment,
* Latent + action-aware expert mesafesi,
* Privileged context’ten training-only teacher router,
* Student gate’e KL distillation.

İlk aşamada şunları eklemezdim:

* Full-output orthogonality,
* Çok agresif entropy loss,
* Sekiz veya daha fazla expert,
* Direct PPO gradient’i student encoder’a,
* Terrain başına kesin bir expert zorlaması.

Önce responsibility mekanizmasının gerçekten expert’ları ayırıp ayırmadığı görülmeli.

Daha sonra gerekirse:

* Residual diversity,
* Temporal route consistency,
* Küçük PPO fine-tuning gradient’i,
* Daha uzun recurrent history

eklenebilir.

## Başarıyı nasıl kanıtlamak gerekir?

Yalnız final latent PCA yeterli değil. Şu ablation’lar gerekli:

* Öğrenilmiş gate yerine uniform gate koyunca performans düşüyor mu?
* Gate yerine random expert seçince düşüyor mu?
* Her expert tek başına hangi terrain/context’lerde iyi?
* Bir expert kapatıldığında hangi rejim zarar görüyor?
* Gate sample entropy’si ve batch marginal entropy’si ne?
* (I(\text{route};\text{terrain/context})) ne kadar?
* Expert residual cosine similarity’si ne?
* Route zaman içinde anlamsız biçimde titriyor mu?
* Parametre sayısı eşleştirilmiş dense encoder’dan daha iyi mi?
* Her expert’ın teacher imitation ve action imitation hatası hangi rejimde düşük?

Eğer uniform gate ablation’ında sonuç değişmiyorsa model hâlâ gerçek bir MoE değildir.

## Net hükmüm

**Dense teacher korunabilir.** Hatta privileged teacher, student expert iş bölümünü öğretmek için güçlü bir avantaja çevrilebilir.

RoboGauge’ın esas hataları:

1. Loss’un yalnız mixed output’a uygulanması.
2. Expert responsibility bulunmaması.
3. Router’a privileged veya semantik supervision verilmemesi.
4. Teacher target’ın hareketli olması.
5. Expert’ların büyük ölçüde ortak trunk paylaşması.
6. Student’ın göremediği teacher bilgisinin L2 ile ortalamaya zorlanması.
7. Latent taklidinin davranış önemini hesaba katmaması.

Ben bu yeni yöntemi kavramsal olarak şöyle adlandırırdım:

**Privileged-Routed Residual MoE-CTS**

Asıl yenilik “bir entropy loss daha eklemek” olmazdı. Asıl yenilik:

> Privileged teacher context ile expert responsibility üretmek, her expert’ı yalnız kendi yerel teacher hedeflerine distill etmek ve student router’a bu responsibility’leri proprioception’dan tahmin ettirmek.

RoboGauge’ı gerçekten kurtaracak değişiklik bu. Bir sonraki mantıklı adım, `StudentMoEEncoder` ve `moe_cts.py` update loop’unu bu objective’e göre doğrudan patchlemek.

[1]: https://github.com/wty-yy/go2_rl_gym/blob/master/rsl_rl/rsl_rl/algorithms/moe_cts.py "go2_rl_gym/rsl_rl/rsl_rl/algorithms/moe_cts.py at master · wty-yy/go2_rl_gym · GitHub"
[2]: https://arxiv.org/html/2604.21330v1 "Teacher-Guided Routing for Sparse Vision Mixture-of-Experts"
[3]: https://arxiv.org/abs/2306.04640 "ModuleFormer: Modularity Emerges from Mixture-of-Experts"
[4]: https://arxiv.org/html/2505.22323v5 "Advancing Expert Specialization for Better MoE"
[5]: https://arxiv.org/html/2402.00893v1 "MoDE: A Mixture-of-Experts Model with Mutual Distillation among the Experts"
