# Upstream MuJoCo MoE-CTS uzmanlaşma sonuçları

Tarih: 2026-08-07  
Kapsam: yayımlanmış upstream deployment-bridge ağırlıkları, resmî `go2_rl_gym` MuJoCo hattı, exact `flat` + `stairs` assetleri.

## Net hüküm

Bu kampanyanın desteklediği seçenek **“diffuse gate ama farklı expert basis”** seçeneğidir. İki checkpointte de gate sekiz expert üzerinde yaygın kalıyor (`effective experts=6.53` ve `6.62`, `mean max gate=0.288` ve `0.291`); buna karşın expert latentleri ve tek-expert action çıktıları birbirinden işlevsel olarak ayrışıyor. Gate’in command bilgisini taşıdığına dair kanıt zayıf-orta, normalized mixed latent command bilgisini belirgin taşıyor; terrain bilgisinde gate ve latent held-out majority baseline’ını geçmiyor.

Bu sonuç **semantik hard routing veya method win kanıtı değildir**. Learned route, bu MuJoCo protokolünde çoğu zaman uniform/top-1’dan daha düşük tracking error verir ve fixed-expert route’lar belirgin biçimde kötüleşir; ancak 164k’da top-1 altındaki fall rate `%0` ile learned `%33.3` olduğundan route causal value tek yönlü değildir. Action MSE’ler aynı-state müdahale ölçüsüdür, closed-loop performans metriği değildir.

## Checkpoint, parity ve provenance

| etiket | bridge checkpoint | boyut | SHA256 | loader schema | teacher/critic |
|---|---|---:|---|---|---|
| 137k | `logs/go2_moects/wty_go2_moe_cts_137k/model_0.pt` | 7,552,692 B | `9dab9f0776510301280b62f3885e3a68e36a7266261adfc9d36023eb3337f27a` | `deployment_bridge` | unavailable |
| 164k | `logs/go2_moects/wty_go2_moe_cts_high_slope_thre_164k/model_0.pt` | 7,552,692 B | `2b1ebf3d8aa9a48bf998555a5157f573d063a13ee38f932ff73bca71e1d66f50` | `deployment_bridge` | unavailable |

Bridge içindeki `infos.provenance`, dosyaların actor + student MoE ağırlıklarından oluştuğunu ve teacher/critic’in yalnız rastgele initialization olarak bulunduğunu bildiriyor. Bu nedenle teacher veya privileged oracle üretilmedi; `teacher_available=false`, `critic_available=false` ve `privileged_obs_source=unavailable_not_fabricated` sınırı korunuyor.

Bridge ağırlıkları, kaynak TorchScript deployment dosyalarıyla mevcut parity API’sindeki `run_jit_parity(adapter=bridge, checkpoint=reference_jit)` çağrısıyla karşılaştırıldı:

| etiket | reference JIT | reference SHA256 | seed | status | max abs action error | mean abs action error |
|---|---|---|---:|---|---:|---:|
| 137k | `go2_rl_gym/deploy/pre_train/go2/go2_moe_cts_137k_0.6739.pt` | `2206659a6a23886446dcfc05d86888d6cdc9b8a4e31f44cbbc4e2d17ba3d6ead` | 17 | PASS | 0.0 | 0.0 |
| 164k | `go2_rl_gym/deploy/pre_train/go2/go2_moe_cts_high_slope_thre_164k_0.6715.pt` | `9d9ad783a1017b6eced5984eb95279cc5b36db8cc84d21e646f46ba2a8023d9d` | 17 | PASS | 0.0 | 0.0 |

Parity çıktıları:

- [137k JIT parity metrics](logs/eval/upstream_moe_cts/wty_go2_moe_cts_137k/jit_parity/metrics.json)
- [164k JIT parity metrics](logs/eval/upstream_moe_cts/wty_go2_moe_cts_high_slope_thre_164k/jit_parity/metrics.json)

Bu parity yalnız deploy ABI/action eşitliğidir; teacher/critic kullanılabilirliği hakkında olumlu kanıt değildir.

## Sabit protokol

- `reference-root=go2_rl_gym`
- Exact terrain assetleri yalnız `flat.xml` ve `stairs.xml`; `wave`/`obstacle` proxy’leri ana kampanyaya alınmadı.
- Asset SHA256: `flat.xml = 0ad2696c6c53701dbba34f1a9a2158e1c1e4f491eb114263b0f86ecadc9e40ac`; `stairs.xml = d3e43ab7e11bf7a7320c45a5cbd42fd5991bca813452cb026da28cc8cf5ce922`.
- Command bank: `paper6 = forward, backward, strafe_left, strafe_right, turn_left, turn_right`.
- `duration_s=5.0`, `simulation_dt=0.002`, `control_decimation=10`, policy period `0.02 s`, `seed=17`.
- Closed-loop route’lar: `learned`, `uniform`, `top1`, `fixed_expert_0` … `fixed_expert_7`.
- Her terrain-command-route kombinasyonu taze MuJoCo `MjModel/MjData` ve taze 5-frame history ile başladı; fall sonrası satır yazılmadı.
- Her checkpoint: `2 terrain × 6 command × 11 route = 132 rollout`; toplam 264 rollout.
- 137k: 18,075 recorded rows; 17,530 valid specialization rows. 164k: 19,948 recorded rows; 19,147 valid specialization rows.
- Fiziksel runaway sırasında finite fakat anlamsız büyüklükte satırlar görüldü. Representation/action specialization istatistikleri tüm gerekli numeric alanların finite olması ve `max_abs_obs <= 1000` filtresiyle üretildi; fall/survival/tracking özetleri tüm route kayıtlarını kullanıyor.

Makine-okunur ana çıktılar:

- [137k closed-loop NPZ](logs/eval/upstream_moe_cts/wty_go2_moe_cts_137k/closed_loop_exact_flat_stairs/probe.npz) ve [metrics](logs/eval/upstream_moe_cts/wty_go2_moe_cts_137k/closed_loop_exact_flat_stairs/metrics.json)
- [164k closed-loop NPZ](logs/eval/upstream_moe_cts/wty_go2_moe_cts_high_slope_thre_164k/closed_loop_exact_flat_stairs/probe.npz) ve [metrics](logs/eval/upstream_moe_cts/wty_go2_moe_cts_high_slope_thre_164k/closed_loop_exact_flat_stairs/metrics.json)

`shuffled` aynı-state intervention olarak tüm closed-loop kayıt bankası üzerinde global seed-17 gate permutation ile hesaplandı; hiçbir fiziksel route shuffled action ile sürülmedi. Bu, `N=1` per-step shuffle’ın yanlışlıkla identity olmasını önler.

## Gate ve expert uzmanlaşması

### Gate yoğunluğu

| checkpoint | entropy mean | effective experts mean | mean max gate | marginal usage (expert 0…7) |
|---|---:|---:|---:|---|
| 137k | 1.8312 | 6.5287 | 0.2884 | `[0.1335, 0.1301, 0.1427, 0.1728, 0.0921, 0.0993, 0.1369, 0.0926]` |
| 164k | 1.8511 | 6.6238 | 0.2912 | `[0.1341, 0.1260, 0.1196, 0.1381, 0.0914, 0.1063, 0.1780, 0.1064]` |

164k, 137k’ya göre biraz daha diffuse: effective-expert sayısı `+0.0952`, entropy `+0.0199`, mean-max gate `+0.0028` (daha keskin değil). 164k’da expert 6 marjinal kullanımı `%17.8` ile en yüksek; yine de tek bir expert baskın değil.

### Expert latent ve action ayrışması

Expert latent normları ham, L2-normalization öncesi değerlerdir; bu nedenle `1e6` ölçeğindeki L2 değerleri action performansı olarak yorumlanmamalıdır. Cosine, norm ölçeğinden bağımsız işlevsel basis ayrışması için daha anlamlıdır.

| ölçü | 137k | 164k |
|---|---:|---:|
| off-diagonal expert cosine mean (min…max) | 0.0707 (`-0.418…0.505`) | 0.1076 (`-0.296…0.496`) |
| off-diagonal raw expert L2 mean (min…max) | 1,057,285 (`294,287…1,497,369`) | 1,352,883 (`925,666…1,838,124`) |
| single-expert action norm mean (expertler arası ortalama) | 13.433 (`12.832…14.112`) | 15.281 (`14.330…15.804`) |
| off-diagonal single-expert action MSE mean (min…max) | 1.7106 (`0.9463…2.1474`) | 1.0966 (`0.6412…1.4918`) |

137k’da action-basis ayrışması 164k’dan daha yüksek; 164k’da cosine ortalaması biraz daha pozitif ve action pairwise MSE daha düşük. Her iki checkpointte de “expert’ler aynı kopya” hipotezi desteklenmiyor.

### Aynı-state action intervention gap’leri

| checkpoint | learned–uniform MSE | learned–shuffled MSE | learned–top1 MSE |
|---|---:|---:|---:|
| 137k | 0.2000 | 0.3381 | 0.3829 |
| 164k | 0.1450 | 0.2241 | 0.2225 |

Bu üç değer aynı observation/history state’inde ağ çıktısının gate müdahalesiyle ne kadar değiştiğini ölçer. Bunları daha düşük/daha yüksek closed-loop kalite diye yorumlamak yanlış olur; closed-loop karşılığı aşağıdaki route tablosundadır.

## Group-stratified command/terrain probe

Sınıflandırma, command/terrain rollout gruplarını train/test arasında ayıran deterministic group-stratified split ve nearest-centroid classifier ile yapıldı. Balanced accuracy ve majority baseline birlikte raporlanıyor.

| checkpoint | temsil | command balanced acc | command majority | terrain balanced acc | terrain majority |
|---|---|---:|---:|---:|---:|
| 137k | gate | 0.2853 | 0.2403 | 0.4520 | 0.5166 |
| 137k | normalized mixed latent | 0.7014 | 0.2196 | 0.3926 | 0.5383 |
| 164k | gate | 0.3366 | 0.2244 | 0.4786 | 0.5385 |
| 164k | normalized mixed latent | 0.7898 | 0.2129 | 0.4099 | 0.5188 |

Command açısından gate, majority’nin biraz üzerinde; normalized mixed latent command bilgisini belirgin taşıyor. Terrain açısından iki temsil de majority baseline’ının altında; bu bankada terrain-semantic gate iddiası desteklenmiyor. Tüm dört probe `available=true`; teacher/oracle probe ise privileged observation bulunmadığı için bilinçli olarak `available=false`.

## Closed-loop sonuçları

Tracking error, her policy adımındaki `[vx, vy, yaw_rate] - command` vektör normunun rollout ortalamasıdır. Achieved velocity sütunu altı command ve iki terrain üzerindeki ortalama `[vx, vy, yaw_rate]` vektörüdür; command’lerin işaretleri birbirini götürebileceği için tek başına başarı skoru değildir.

### 137k

| route | fall rate | survival (s) | tracking error | achieved `[vx,vy,wz]` |
|---|---:|---:|---:|---|
| learned | 0.333 | 4.383 | 0.334 | `[-0.002,-0.045,0.002]` |
| uniform | 0.333 | 4.620 | 0.606 | `[-0.002,-0.008,-0.067]` |
| top1 | 0.333 | 4.343 | 0.623 | `[0.038,-0.068,-0.002]` |
| fixed_expert_0 | 0.833 | 2.240 | 1.835 | `[-0.595,0.843,0.014]` |
| fixed_expert_1 | 0.583 | 3.215 | 2.077 | `[0.875,0.578,-0.077]` |
| fixed_expert_2 | 1.000 | 2.803 | 2.313 | `[0.846,-0.738,0.015]` |
| fixed_expert_3 | 0.500 | 3.633 | 2.004 | `[-0.849,-0.788,-0.301]` |
| fixed_expert_4 | 1.000 | 1.330 | 1.926 | `[0.001,0.049,0.689]` |
| fixed_expert_5 | 1.000 | 1.123 | 1.493 | `[-0.042,-0.217,0.054]` |
| fixed_expert_6 | 1.000 | 1.550 | 1.686 | `[0.159,0.261,0.679]` |
| fixed_expert_7 | 1.000 | 0.883 | 2.266 | `[-0.026,0.169,-1.576]` |

Learned, uniform ve top1 aynı fall rate’e sahip. Learned tracking error uniform’a göre `-0.272`, top1’a göre `-0.289`; survival learned’ın uniform’dan `-0.237 s`, top1’dan `+0.040 s` farkında. Fixed expert’ların tamamı learned’dan daha yüksek tracking error’a, `+0.167…+0.667` fall-rate farkına ve `0.883…3.633 s` survival’a sahip.

### 164k

| route | fall rate | survival (s) | tracking error | achieved `[vx,vy,wz]` |
|---|---:|---:|---:|---|
| learned | 0.333 | 4.547 | 0.386 | `[-0.019,-0.019,0.072]` |
| uniform | 0.333 | 4.573 | 0.481 | `[0.050,-0.052,-0.010]` |
| top1 | 0.000 | 5.000 | 0.566 | `[-0.015,0.025,-0.023]` |
| fixed_expert_0 | 0.667 | 3.540 | 1.732 | `[-0.848,0.209,-0.122]` |
| fixed_expert_1 | 0.833 | 2.620 | 1.991 | `[0.131,-1.043,-0.035]` |
| fixed_expert_2 | 1.000 | 1.520 | 1.969 | `[0.124,0.660,-0.308]` |
| fixed_expert_3 | 0.417 | 4.290 | 1.714 | `[0.871,0.092,-0.029]` |
| fixed_expert_4 | 1.000 | 0.487 | 1.017 | `[-0.080,-0.081,0.107]` |
| fixed_expert_5 | 1.000 | 1.373 | 1.636 | `[-0.072,0.252,0.919]` |
| fixed_expert_6 | 0.583 | 3.393 | 1.215 | `[0.045,0.133,0.074]` |
| fixed_expert_7 | 0.833 | 1.903 | 2.085 | `[-0.065,-0.007,-1.031]` |

Learned tracking error uniform’a göre `-0.096`, top1’a göre `-0.180`; survival learned’ın uniform’dan `-0.027 s`, top1’dan `-0.453 s` farkında. Top1 burada fall rate’i `%0` ve survival’ı `5.0 s` yaparken tracking error learned’dan daha kötü; bu, routing causal value’nun tek bir scalar ile özetlenemediğini gösteriyor. Fixed expert’lar yine learned’dan belirgin kötü tracking/survival profiline sahip; en iyi fixed survival `fixed_expert_3` (`4.29 s`) olsa da tracking error `1.714` ve fall rate `%41.7`.

## 137k–164k karşılaştırması

1. **Gate concentration:** 164k daha keskin değil, tersine biraz daha diffuse (`effective 6.624 vs 6.529`, `entropy 1.851 vs 1.831`, `mean max 0.291 vs 0.288`).
2. **Command information:** Gate balanced accuracy `0.285 → 0.337`; normalized mixed latent `0.701 → 0.790`. Her iki temsil de command bilgisini 164k’da daha iyi taşıyor; gate kazanımı majority’ye göre yine sınırlı.
3. **Terrain information:** Gate `0.452 → 0.479`, fakat majority `0.517 → 0.538`; latent `0.393 → 0.410`, majority `0.538 → 0.519`. Hiçbirinde terrain classifier majority’yi aşmadı.
4. **Functional expert diversity:** 137k off-diagonal cosine `0.0707`, single-expert action MSE `1.7106`; 164k cosine `0.1076`, action MSE `1.0966`. 164k’da basis cosine olarak biraz daha benzer, action ayrışması olarak daha düşük; 137k functional diversity sinyali daha güçlü.
5. **Routing causal value:** Her iki modelde fixed expert route’lar çoğunlukla çöküyor ve learned route tracking error’da daha iyi. Ancak 164k top1 survival’da learned’ı geçiyor; bu, learned gate’in her performans ekseninde üstün olduğunu kanıtlamıyor.

## Genesis 23.5k ile yalnız cross-simulator bağlam

Daha önce ölçülen Genesis 23.5k değerleri referans olarak: learned/uniform/shuffled/top1/oracle action gap `0.565/0.774/1.001/7.869/1.894`, effective expert `~7.68`, mean max gate `~0.180`, functional expert cosine `~0.28–0.30`. Bu kampanya MuJoCo’da farklı state/history dağılımı, farklı terrain/route toplama ve farklı finite-row sınırı kullanıyor; dolayısıyla sayılar doğrudan eşleştirilmiş değil ve 137k/164k için method win iddiası kurmakta kullanılamaz.

## Hipotez, belirsizlik ve sonraki ayırt edici deney

### Hipotez

Upstream policy’de gate, sekiz farklı function basis’i diffuse biçimde karıştıran conditional ensemble gibi davranıyor. Komut bilgisi özellikle normalized mixed latent’te temsil ediliyor; gate’in kendisi bu bilgiyi zayıf taşıyor. Terrain, bu 5 s MuJoCo bankasında gate/latent için gözlenebilir bir routing ekseni olarak görünmüyor. 164k checkpoint command decoding’i güçlendirmiş, fakat gate’i sparse/semantic hâle getirmemiş.

### Belirsizlik

- Her checkpoint tek seed ve tek deployment artifact; 3-seed varyans ölçülmedi.
- Bridge’de teacher/critic gerçek eğitim ağırlıkları yok; privileged teacher/oracle karşılaştırması yapılamaz.
- MuJoCo `flat` + `stairs` exact assetleri kullanıldı; upstream eğitim terrain bankasının tamamı değildir.
- Fixed-expert route’lar off-policy müdahalelerdir ve bazıları fiziksel olarak kararsızdır; onların başarısızlığı tek başına expert’in “yanlış semantiği” olduğunu kanıtlamaz.
- Command/terrain probe group-stratified ve held-out’tur, fakat tek nearest-centroid probe source causality kanıtı değildir.
- 5 s route’lar ve height-based fall termination uzun-horizon robustness yerine kısa-horizon fiziksel farkı ölçer.

### Sonraki ayırt edici deney

Her iki checkpoint için aynı frozen MuJoCo state/history bankasında üç bağımsız seed ve episode/command/terrain group holdout kullanılsın. Aynı state üzerinde command’i counterfactual olarak değiştirip gate, normalized latent ve action’ı ayrı ölçmek; ayrıca 20 s closed-loop route’larda learned/uniform/top1/fixed expert fall-survival-tracking üçlüsünü tekrar etmek şu ayrımı netleştirir: command semantiği gate’in kendisinde mi, yalnız mixed latent’te mi; ve functional basis farkı uzun-horizon robustness’a taşınıyor mu? Bu deneyde minimum karar kapısı, gate’in majority baseline üstü held-out command balanced accuracy’si ve learned route’un fixed/uniform/top1’a karşı seed-robust task avantajıdır; entropy veya action MSE tek başına geçer notu değildir.

## Kod/test notu

Deney sırasında üç gerçek altyapı sorunu dar kapsamda düzeltildi ve testle doğrulandı: local `history_encoder` bridge’inin yanlışlıkla `source_training` etiketlenmesi; closed-loop `N=1` yüzünden shuffled intervention’ın identity kalması; finite fakat fiziksel runaway satırlarının specialization istatistiklerini bozması. `tests/test_probe_upstream_moe_cts.py`: **16 passed**. Kullanıcı/ajan çalışma ağacındaki diğer değişiklikler korunmuştur; commit/stage/revert yapılmadı.
