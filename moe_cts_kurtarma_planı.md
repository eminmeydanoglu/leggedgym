# MoE-CTS Kurtarma Planı

**Durum:** Uygulama bekliyor
**Tarih:** 2026-08-06
**Korunacak baseline:** `go2_moects` / `ActorCriticMoECTS` / `PPO_MOE_CTS`
**Birincil kaynak koşu:** `logs/go2_moects/Aug03_12-01-45_moe_cts_genesis`
**Birincil başlangıç checkpoint'i:** `model_23500.pt`
**Birincil probe bankası:** `logs/eval/moe_gate_probe/Aug03_12-01-45_moe_cts_genesis/model_7500/samples.pt`

Bu belge, mevcut MoE-CTS implementasyonunu tek seferde yeniden yazmak için değil, en ucuz ayırt edici deneyden başlayarak aşamalı biçimde kurtarmak veya erken öldürmek için hazırlanmıştır. Her aşama bir önceki aşamanın çıktısına bağlıdır. Kapı geçilmeden sonraki mimari uygulanmayacaktır.

---

## 0. Canlı takip tablosu

Bu tablo her implementasyon/deney turundan sonra aynı commit içinde güncellenecektir. İzin verilen durumlar:

- `NOT_STARTED`
- `IN_PROGRESS`
- `PASS`
- `PARTIAL`
- `STOP`
- `BLOCKED`

| Kimlik | İş paketi | Durum | Commit/run | Kanıt yolu | Karar/not |
|---|---|---|---|---|---|
| `P0-A` | Baseline manifest ve hash'ler | `NOT_STARTED` | - | - | - |
| `P0-B` | 7500/15000/23500 tekrar üretilebilir metrikler | `NOT_STARTED` | - | - | - |
| `P1-A` | Probe V2 schema implementasyonu | `NOT_STARTED` | - | - | - |
| `P1-B` | Collection seed 1 bankası | `NOT_STARTED` | - | - | - |
| `P1-C` | Collection seed 2 bankası | `NOT_STARTED` | - | - | - |
| `P1-D` | Episode/env/physics-cell leakage kontrolü | `NOT_STARTED` | - | - | - |
| `P2-A` | Saf loss/component API ve unit testler | `NOT_STARTED` | - | - | - |
| `P2-B` | V1 banka offline pilot | `NOT_STARTED` | - | - | - |
| `P2-C` | V2 banka confirmatory offline deney | `NOT_STARTED` | - | - | - |
| `P2-D` | `PASS / PARTIAL / STOP` kararı | `NOT_STARTED` | - | - | - |
| `P3-A` | Action-aware offline deney | `NOT_STARTED` | - | - | Yalnız P2 `PARTIAL` ise |
| `P4-A` | `go2_moects_local` implementasyonu | `NOT_STARTED` | - | - | P2/P3 kapısına bağlı |
| `P4-B` | 4-env ve 64-env smoke | `NOT_STARTED` | - | - | - |
| `P4-C` | 1024-env medium gate | `NOT_STARTED` | - | - | - |
| `P4-D` | 3-seed 15k campaign | `NOT_STARTED` | - | - | - |
| `P5-A` | Parameter/FLOP-matched dense baseline | `NOT_STARTED` | - | - | - |
| `P5-B` | MoE gereklilik kararı | `NOT_STARTED` | - | - | - |
| `P6-A` | Residual encoder/actor/algorithm | `NOT_STARTED` | - | - | P4 ve P5 geçmeden başlamaz |
| `P6-B` | Offline bootstrap ve export/resume | `NOT_STARTED` | - | - | - |
| `P6-C` | 3-seed residual campaign | `NOT_STARTED` | - | - | - |
| `P7-*` | Opsiyonel routing uzantıları | `NOT_STARTED` | - | - | Her biri ayrı karar |

### 0.1 Her takip güncellemesinde zorunlu alanlar

Bir satır `PASS`, `PARTIAL`, `STOP` veya `BLOCKED` yapılırken şunlar yazılacak:

1. Kod commit'i ve dirty-state.
2. Tam komut veya campaign config yolu.
3. Kaynak checkpoint ve SHA-256.
4. Veri bankası/split manifesti ve SHA-256.
5. Seed listesi.
6. Ham metrik dosyası ve insan-okunur rapor yolu.
7. Kapıdaki her sayısal maddenin sonucu.
8. Kararın yalnız pilot mu, confirmatory mi olduğu.
9. Bilinen belirsizlikler ve bir sonraki ayırt edici kontrol.

### 0.2 Durum geçiş kuralları

- `NOT_STARTED -> IN_PROGRESS`: Kod veya koşu gerçekten başladığında.
- `IN_PROGRESS -> PASS`: Tüm zorunlu kapı maddeleri ve kanıt yolları tamamlandığında.
- `IN_PROGRESS -> PARTIAL`: Yalnız belirli hipotez desteklenmiş, ana kapı geçilmemişse.
- `IN_PROGRESS -> STOP`: Öldürme kriterlerinden biri confirmatory kanıtla gerçekleşmişse.
- `IN_PROGRESS -> BLOCKED`: Eksik veri, altyapı veya dış bağımlılık koşuyu imkânsız kılıyorsa.
- Pilot sonuç `PASS` veremez; en fazla `PARTIAL` verebilir.
- Bir üst aşamanın satırı, bağlı olduğu alt kapı `PASS` olmadan `IN_PROGRESS` yapılmaz.

---

## 1. Net karar

Mevcut sistem tamamen işlevsiz değildir. Learned soft mixture, uniform ve shuffled mixture'dan daha iyi teacher-action taklidi yapmaktadır. Buna karşın sistem, tek başına yetkin ve belirli rejimlere sahip expert'lar öğrenmemiştir. Mevcut yapı en doğru biçimde **dense conditional basis encoder** olarak tanımlanabilir.

Kurtarma stratejisi:

1. Mevcut mixed latent objective korunacak.
2. Önce frozen/offline bir deneyle, mixed objective'in yanına düşük ağırlıklı local responsibility eklenecek.
3. Bu deney bireysel expert yetkinliği üretmezse semantic MoE hattı kapatılacak.
4. Offline deney geçerse aynı objective, ayrı bir `go2_moects_local` task'ında on-policy olarak denenecek.
5. On-policy minimal yol da geçerse shared-base + independent residual experts mimarisi `go2_moects_residual` adıyla uygulanacak.
6. Sinkhorn, privileged router, EMA teacher ve sparse top-2 ilk implementasyona girmeyecek. Bunlar ayrı kapılara bağlı opsiyonel uzantılardır.

Kısa ifadeyle: **önce objective sinyalini izole et, sonra mimariyi değiştir. İkisini aynı deneyde değiştirme.**

---

## 2. Mevcut kanıt ve başlangıç hipotezleri

### 2.1 Ölçülen bulgular

`model_7500` training-fidelity bankasında:

| Metrik | Değer |
|---|---:|
| Ham gate effective experts | `7.46 / 8` |
| Norm-ağırlıklı effective experts | `6.04 / 8` |
| Ortalama en büyük gate ağırlığı | `0.209` |
| Expert functional off-diagonal cosine | `0.278` |
| Learned action gap | `0.472` |
| Uniform action gap | `0.752` |
| Shuffled action gap | `1.052` |
| Top-1 action gap | `6.941` |
| Tek-expert latent-oracle action gap | `1.884` |

Aynı bankanın salt-okunur daha geç checkpoint analizinde:

| Checkpoint | Effective experts | Mean max gate | Learned gap | Uniform gap | Top-1 gap | Oracle gap |
|---|---:|---:|---:|---:|---:|---:|
| `7500` | `7.46` | `0.209` | `0.472` | `0.752` | `6.941` | `1.884` |
| `15000` | `7.67` | `0.182` | `0.542` | `0.764` | `8.321` | `2.055` |
| `23500` | `7.68` | `0.180` | `0.565` | `0.774` | `7.869` | `1.894` |

Bu tablo şunları söyler:

- Gate zamanla daha keskin değil, daha uniform hale gelmektedir.
- Expert'lar birbirinin kopyası değildir.
- Learned gate gerçek girdi-bağımlı bilgi taşımaktadır.
- Mevcut expert'lar tek başına politika için yeterli latent üretmemektedir.
- Bugün hard top-1/top-2'ye geçmek bir routing ayarı değil, temsil sınıfını değiştirmektir.

### 2.2 Başlangıç hipotezleri

Plan şu hipotezleri ayrı ayrı test edecektir:

| Kimlik | Hipotez |
|---|---|
| `H1` | Mixed-output loss local expert kredi sinyali vermediği için expert'lar basis bileşenlerine dönüşüyor. |
| `H2` | Düşük ağırlıklı soft local responsibility, mevcut mixed performansı bozmadan bireysel expert yetkinliği oluşturabilir. |
| `H3` | Yalnız latent cost yeterli olmayabilir; actor-duyarlı action cost gerekebilir. |
| `H4` | Terrain semantiği mevcut history'den yeterince gözlenebilir değildir; command/contact/phase eksenleri daha gerçekçi specialization hedefidir. |
| `H5` | Mevcut MoE avantajı routing'den değil ek parametre ve dense basis kapasitesinden geliyor olabilir. |
| `H6` | Full residual mimari, local responsibility sinyali kanıtlanmadan uygulanırsa aynı belirsizliği daha pahalı biçimde tekrarlar. |

---

## 3. Değişmezler ve regresyon duvarı

Aşağıdaki sözleşmeler tüm aşamalarda korunacaktır:

1. `go2_moects` task'ının davranışı, config'i ve checkpoint ABI'si değişmeyecek.
2. Mevcut kaynak-faithful loss, yeni katsayılar sıfırken bit-eşdeğer veya kabul edilen kayan nokta toleransında eşdeğer kalacak.
3. Student encoder PPO surrogate gradyanı almayacak.
4. Critic loss teacher veya student encoder'a gradyan vermeyecek.
5. Teacher latent student loss içinde `stopgrad` kalacak.
6. Teacher/student interleaved env mapping, storage ordering ve advantage normalization değişmeyecek.
7. Actor input ABI'si `[latent, obs]` olarak kalacak.
8. Export edilen student policy girişi `(obs, history)` olarak kalacak.
9. Mevcut `history_encoder_optimizer_state_dict` resume sözleşmesi korunacak.
10. `go2_moects_him` ve diğer CTS kolları etkilenmeyecek.
11. Probe ve offline script'ler hiçbir kaynak checkpoint'i overwrite etmeyecek.
12. Rastgele sample split bilimsel doğrulama olarak kabul edilmeyecek.

---

## 4. Aşama haritası

| Aşama | Amaç | Çıktı | Sonraki kapı |
|---|---|---|---|
| `P0` | Baseline'ı ve veri sözleşmesini dondur | Manifest + tekrar üretilebilir metrikler | Veri yeterli mi? |
| `P1` | Provenance-rich probe bankası topla | `samples_v2.pt` + split manifest | Leakage yok mu? |
| `P2` | Frozen offline soft-local fine-tune | Ayrı checkpoint ve rapor | Bireysel expert yetkinliği oluştu mu? |
| `P3` | Action-aware offline cost | Latent-only vs action-aware karşılaştırma | Davranış maliyeti gerekli mi? |
| `P4` | Additive on-policy minimal objective | `go2_moects_local` | Return/robustness korunuyor mu? |
| `P5` | Parameter/FLOP-matched dense baseline | Adil kapasite karşılaştırması | MoE gerekli mi? |
| `P6` | Full residual MoE | `go2_moects_residual` | Yeni mimari faydalı mı? |
| `P7` | Opsiyonel routing uzantıları | Sinkhorn/privileged/EMA/top-2 | Her biri ayrı ablation |

`P2` başarısızsa `P4-P7` uygulanmaz. `P4` başarısızsa `P6-P7` uygulanmaz.

---

## 5. P0: Baseline dondurma ve tekrar üretilebilirlik

### 5.1 Dondurulacak kaynaklar

- Run: `Aug03_12-01-45_moe_cts_genesis`
- Checkpoint'ler: `7500`, `15000`, `23500`
- Probe bankası: mevcut 35.328 sample
- Kaynak model state hash'leri
- Aktif Git commit ve dirty-state manifesti
- Effective env/policy/algorithm config'i
- Probe script sürümü

### 5.2 Yeni baseline manifesti

Yeni dosya:

`logs/eval/moe_specialization/baseline_manifest.json`

Zorunlu alanlar:

```json
{
  "schema_version": 1,
  "source_run": "Aug03_12-01-45_moe_cts_genesis",
  "checkpoints": {
    "7500": {"path": "...", "sha256": "..."},
    "15000": {"path": "...", "sha256": "..."},
    "23500": {"path": "...", "sha256": "..."}
  },
  "sample_bank": {"path": "...", "sha256": "...", "num_samples": 35328},
  "git_commit": "...",
  "git_dirty": true,
  "metrics_version": "moe_specialization_v1"
}
```

### 5.3 Baseline metrikleri

Her checkpoint için tek bir `metrics_pre.json` üretilecek:

- Mixed latent MSE ve cosine
- Teacher/student action gap
- Uniform, shuffled, top-1, top-2 action gap
- Latent-oracle tek-expert gap
- Action-oracle tek-expert gap
- Expert başına latent/action gap
- Leave-one-expert-out latent ve action etkisi
- Ham gate entropy/effective experts/max weight
- Norm-ağırlıklı katkı metrikleri
- Expert output norm dağılımı
- Functional cosine ve linear CKA
- Gate/assignment ile command/contact/terrain/physics ilişkisi

### 5.4 P0 kabul kapısı

- Aynı bankada aynı checkpoint iki kez analiz edildiğinde tüm deterministik metrikler `rtol=1e-6`, `atol=1e-7` içinde eşleşmeli.
- Probe output dizini checkpoint'ten bağımsız verilebilmeli; başka checkpoint analizi eski `results.json` dosyasını overwrite etmemeli.
- Manifest hash'leri doğrulanmadan offline fine-tune başlamamalı.

---

## 6. P1: Provenance-rich probe bankası

Mevcut banka hızlı pilot için yeterlidir, fakat confirmatory split için eksiktir. Kayıtlı alanlar yalnız `obs`, `obs_history`, `privileged_obs`, `commands`, `base_lin_vel`, `base_ang_vel`, `terrain_id` ve `terrain_level` alanlarıdır. Episode, env ve canlı physics readback kimliği yoktur.

### 6.1 Mevcut bankanın kullanım sınırı

Mevcut veriyle pilot split şöyle kurulabilir:

- `record_idx = row // num_envs`
- `env_id = row % num_envs`
- Env'ler terrain sınıfına göre stratified biçimde train/validation/test'e ayrılır.
- Ayrı bir temporal-block split uygulanır.

Bu split yalnızca **pilot** sayılır. Episode ve DR-cell leakage ihtimali nedeniyle `P2 PASS` kararı veremez.

### 6.2 `samples_v2.pt` şeması

`probe_moe_gate.py --mode collect` aşağıdaki alanlarla genişletilecek:

| Alan | Tip/şekil | Amaç |
|---|---|---|
| `env_id` | `int32 [N]` | Fiziksel env grubu |
| `episode_id` | `int64 [N]` | Env-bağımsız benzersiz episode kimliği |
| `episode_step` | `int32 [N]` | Episode içi zaman |
| `control_step` | `int64 [N]` | Global kontrol saati |
| `done` | `bool [N]` | Boundary tespiti |
| `contact_bits` | `uint8 [N,4]` | Contact/phase proxy |
| `foot_force_z` | `float32 [N,4]` | Temas rejimi |
| `friction` | `float32 [N]` | Canlı readback |
| `added_mass` | `float32 [N]` | Canlı readback |
| `com_displacement` | `float32 [N,3]` | Canlı readback |
| `motor_strength` | `float32 [N,12]` veya summary | Canlı readback |
| `motor_zero_offset` | `float32 [N,12]` veya summary | Canlı readback |
| `control_delay` | `int8 [N]` | Canlı delay draw |
| `pd_gain_scale` | `float32 [N]` | Mevcutsa canlı readback |
| `physics_cell_id` | `int64 [N]` | Kuantize physics tuple hash'i |
| `collection_seed` | `int32 [N]` | Banka provenance'i |

Readback'i doğrulanmamış eksen sessizce sıfırla doldurulmayacak. Alan ya gerçek değerle yazılacak ya da manifestte `unavailable` olarak işaretlenecek.

### 6.3 Split sözleşmesi

Üç ayrı split raporlanacak:

1. **Episode holdout:** Aynı `episode_id` yalnız bir split'te bulunur.
2. **Env holdout:** Aynı `env_id` yalnız bir split'te bulunur.
3. **Physics-cell holdout:** Test physics cell'leri train'de bulunmaz.

Varsayılan oran:

- Train `%60`
- Validation `%20`
- Test `%20`

Terrain ve command-bin kapsaması split'ler arasında raporlanacak; coverage eksikse split fail edecektir.

### 6.4 Toplama kapsamı

- En az iki collection seed
- Her aktif terrain ailesi için en az `5000` sample
- `stepping_stones` ve `gap` layout'ta hiç yoksa confirmatory banka tamamlanmış sayılmaz; ya layout düzeltilir ya da deney kapsamından açıkça çıkarılır.
- Nominal ve training-fidelity DR bankaları ayrı tutulur.
- Confirmation kararı training-fidelity test split'i üzerinden verilir.

### 6.5 P1 testleri

Yeni test dosyası:

`tests/test_moe_probe_schema.py`

Testler:

- Tüm zorunlu key'ler aynı `N` boyutuna sahip.
- `episode_id` done sonrası değişiyor.
- Split'ler arası episode/env/physics-cell kesişimi boş.
- Aynı seed aynı split manifestini üretiyor.
- Eksik readback fail-closed davranıyor.
- Eski v1 banka analiz edilebiliyor fakat confirmation olarak işaretlenmiyor.

---

## 7. P2: Frozen offline soft-local deney

Bu aşama planın en ucuz ve en ayırt edici deneyidir. Amaç yeni bir policy eğitmek değil, mevcut expert head'lerinin local kredi sinyali altında bireysel olarak yetkinleşip yetkinleşemediğini test etmektir.

### 7.1 Yeni dosyalar

| Dosya | Sorumluluk |
|---|---|
| `rsl_rl/utils/moe_specialization.py` | Cost, responsibility, KL, balance ve metrik saf tensor fonksiyonları |
| `legged_gym/scripts/eval/moe_specialization_offline.py` | Offline train/eval CLI |
| `configs/eval/moe_specialization_offline.yaml` | Deney matrisi ve varsayılanlar |
| `tests/test_moe_specialization_utils.py` | Loss/gradient matematiği |
| `tests/test_moe_specialization_offline.py` | Split, checkpoint ve CLI sözleşmesi |

### 7.2 Mevcut modülde additive API

`rsl_rl/modules/moe_utils.py` davranış değiştirmeden şu API'lerle genişletilecek:

```python
class Experts(nn.Module):
    def forward_features(self, x): ...
    def forward_heads(self, features): ...
    def forward(self, x): ...  # eski davranış aynı

class MoE(nn.Module):
    def forward_components(self, x):
        # features, expert_outputs, gate_logits, weights, mixed_raw
        ...

class StudentMoEEncoder(nn.Module):
    def forward_components(self, obs): ...
    def forward(self, obs): ...  # export ABI aynı
    def forward_with_weights(self, obs): ...  # telemetry ABI aynı
```

`forward()` ve `forward_with_weights()` yeni component yolunu kullansa bile eski çıktıyla aynı olmalıdır.

### 7.3 Head-only local gradient

Mixed branch:

```python
features = experts.forward_features(history)
expert_outputs = experts.forward_heads(features)
mixed = normalize(sum(weights * expert_outputs))
```

Local branch:

```python
local_expert_outputs = experts.forward_heads(features.detach())
```

Böylece:

- `L_mix` expert backbone, heads ve router'ı günceller.
- `L_local` yalnız ayrık grouped expert heads'i günceller.
- `L_router` yalnız gating network'ü günceller.
- Local responsibility ortak backbone'u dolaylı biçimde bütün expert'lar için değiştirmez.

### 7.4 Loss tanımı

Teacher target:

\[
t_i = \operatorname{stopgrad}(\operatorname{Norm}(E_T(p_i))).
\]

Mevcut mixed latent:

\[
z_i = \operatorname{Norm}\left(\sum_k g_{ik}e_{ik}\right).
\]

Korunacak ana loss:

\[
L_{mix}=\frac{1}{B}\sum_i \lVert z_i-t_i\rVert_2^2.
\]

Normalize expert candidate:

\[
\hat e_{ik}=\operatorname{Norm}(e_{ik}).
\]

Latent cost:

\[
C^z_{ik}=1-\cos(\hat e_{ik},t_i).
\]

Gate prior kullanmayan ilk responsibility:

\[
q^0_{ik}=\operatorname{softmax}_k(-C^z_{ik}/\tau).
\]

Smoothing:

\[
q_{ik}=\operatorname{stopgrad}\left((1-\epsilon)q^0_{ik}+\epsilon/K\right).
\]

Local expert loss:

\[
L_{local}=\frac{1}{B}\sum_{i,k}q_{ik}C^z_{ik}.
\]

Router imitation:

\[
L_{route}=\frac{1}{B}\sum_i KL(q_i\Vert g_i).
\]

Balance:

\[
L_{balance}=\frac{1}{K}\sum_k\left(\frac{1}{B}\sum_i g_{ik}-\frac{1}{K}\right)^2.
\]

Toplam:

\[
L=L_{mix}+\eta L_{local}+\lambda_rL_{route}+\lambda_bL_{balance}.
\]

### 7.5 Kesin `stop-gradient` sınırları

| Yol | Gradyan alacak parametreler | Gradyan almayacaklar |
|---|---|---|
| `L_mix` | Expert backbone, expert heads, router | Teacher encoder, actor, critic |
| `L_local` | Expert heads | Expert backbone, router, teacher, actor, critic |
| `L_route` | Router | Expert backbone/heads, teacher, actor, critic |
| `L_balance` | Router | Diğer tüm parametreler |

`q` her durumda detached olacaktır. Router, cost'u manipüle ederek kendi pseudo-label'ını değiştiremeyecektir.

### 7.6 İlk hyperparameter uzayı

| Parametre | İlk değer/uzay | Not |
|---|---|---|
| `tau_mode` | `target_max_responsibility` | Ham sabit tau yerine bankaya kalibre |
| `target_mean_max_q` | `{0.35, 0.45}` | Probe'da `tau~0.35` yaklaşık `0.38` veriyor |
| `epsilon_q` | `0.03` | Dead expert'a karşı küçük taban |
| `local_grad_ratio_target` | `{0.05, 0.10}` | Ham `eta` yerine gradient RMS oranı |
| `router_loss_coef` | `{0.1, 0.3}` | Router'ı bir anda sertleştirmeyecek |
| `load_balance_coef` | `0.01` | Mevcut değer; ayrı ablation |
| `gate_prior_beta` | `0.0` | İlk deneyde mevcut gate pseudo-label'a girmez |
| `batch_size` | `4096` | Banka karışık olacak kadar büyük |
| `max_epochs` | `30` | Early stopping zorunlu |
| `patience` | `5` | Validation mixed/action gap |
| `head_lr` | `{1e-4, 3e-4}` | Fine-tune, scratch değil |
| `router_lr` | `{1e-4, 3e-4}` | Ayrı param group |
| `weight_decay` | `{0, 1e-5}` | Norm patlaması kontrolü |

Tam kartezyen grid çalıştırılmayacak. Önce küçük pilot:

- `mix_only` kontrolü
- `soft_local` dört kombinasyon
- Her kombinasyon için üç optimizer seed'i
- En iyi iki kombinasyon `samples_v2` iki collection seed'i üzerinde tekrar

### 7.7 Negatif kontroller

Aşağıdaki kontroller planlı olarak çalıştırılacak:

1. `mix_only`: Aynı epoch/LR ile yalnız mevcut loss.
2. `unweighted_local`: `mean_k(C_k)`; expert kopyalanması beklenir.
3. `router_only`: Expert'lar frozen, yalnız router KL.
4. `local_no_router`: Local expert loss var, router imitation yok.

Bu kontroller, kazanımın yalnız ekstra gradient step veya daha uzun fine-tune'dan gelmediğini göstermelidir.

### 7.8 Offline output dizini

```text
logs/eval/moe_specialization_offline/
└── <source_run>/
    └── model_<ckpt>/
        └── <trial_id>/
            ├── config_resolved.json
            ├── manifest.json
            ├── split_manifest.json
            ├── metrics_pre.json
            ├── metrics_best_val.json
            ├── metrics_test.json
            ├── offline_model.pt
            ├── offline_optimizer.pt
            ├── history.csv
            ├── REPORT.md
            └── plots/
```

`offline_model.pt` metadata'sı:

```json
{
  "checkpoint_kind": "offline_specialization_only",
  "source_checkpoint_sha256": "...",
  "sample_bank_sha256": "...",
  "objective_version": "soft_local_v1",
  "resume_training_supported": false
}
```

Offline checkpoint normal training resume checkpoint'i gibi kullanılmayacak. Warm-start dönüşümü ayrı ve açık bir script ile yapılacaktır.

### 7.9 Offline metrikler

Her split için:

- `mixed_latent_mse`
- `mixed_latent_cosine`
- `mixed_action_gap`
- `uniform_action_gap`
- `shuffled_action_gap`
- `top1_action_gap`
- `top2_action_gap`
- `latent_oracle_action_gap`
- `action_oracle_action_gap`
- Expert başına action gap ortalama/medyan/p90
- Best-expert margin: ikinci en iyi eksi en iyi cost
- Assignment sample entropy
- Assignment marginal entropy
- Gate sample entropy
- Gate marginal entropy
- `KL(q || g)`
- Expert assignment usage min/max/std
- Dead expert sayısı
- Expert output norm min/max/oran
- Functional cosine/CKA
- Leave-one-expert-out action degradation
- Gaussian history noise altında route agreement
- History frame dropout altında route agreement
- Ardışık kayıtlarda route switch rate
- Terrain, level, `vx`, yaw, contact ve physics cell ile normalized MI

### 7.10 P2 karar kapısı

Bir trial'ın `PASS` olması için validation ve test split'lerinde tüm şartlar:

1. `mixed_action_gap_post <= 1.10 * mixed_action_gap_pre`
2. `mixed_latent_mse_post <= 1.10 * mixed_latent_mse_pre`
3. `action_oracle_gap_post <= 0.70 * action_oracle_gap_pre`
4. `top2_action_gap_post <= 0.70 * top2_action_gap_pre` veya `top2_action_gap_post <= 1.5 * mixed_action_gap_post`
5. En az `K-1` expert assignment mass'i `>= 0.5/K`
6. Hiçbir expert assignment mass'i `> 2.5/K`
7. Noise route agreement `>= 0.70`
8. Train iyileşmesi test iyileşmesinin iki katından fazla olmamalı
9. En az iki collection seed'inde yön tutarlı olmalı
10. En az iki optimizer seed'inde kapı geçilmeli

Karar:

| Sonuç | Eylem |
|---|---|
| Tüm kritik metrikler PASS | `P3` ve sonra `P4` |
| Bireysel latent iyileşiyor, action iyileşmiyor | Yalnız `P3` action-aware |
| Mixed korunuyor, expert yetkinliği artmıyor | Semantic hattı kapat; dense basis olarak devam |
| Mixed belirgin bozuluyor | Local objective'i kapat; `P4-P7` uygulanmaz |
| Yalnız train split iyileşiyor | Overfit; veri/split düzeltilmeden devam edilmez |

---

## 8. P3: Action-aware responsibility

Bu aşama yalnız latent-only local loss bireysel latent yetkinliği oluşturup action gap'i iyileştirmezse uygulanacak.

### 8.1 Action cost

Teacher action:

\[
\mu^T_i=\operatorname{stopgrad}(\pi_\mu(o_i,t_i)).
\]

Expert action:

\[
\mu^k_i=\pi_\mu(o_i,\hat e_{ik}).
\]

Action cost:

\[
C^a_{ik}=\frac{1}{A}\left\lVert\frac{\mu^k_i-\mu^T_i}{s_a}\right\rVert_2^2.
\]

Birleşik cost:

\[
C_{ik}=C^z_{ik}+\rho C^a_{ik}.
\]

`s_a`, action boyutlarının bankadaki teacher standard deviation'ıdır ve minimum epsilon ile clamp edilir.

### 8.2 Actor gradient sözleşmesi

- Actor parametreleri kesinlikle güncellenmeyecek.
- Expert latent'a actor Jacobian'ı üzerinden gradyan geçecek.
- Actor parametrelerinde gradient accumulation oluşmayacak.
- Tercih edilen uygulama `torch.func.functional_call` ile detached actor parametreleridir.
- Geçici `requires_grad_(False/True)` mutasyonu kullanılmayacak; exception durumunda optimizer sözleşmesini bozabilir.

### 8.3 İlk sweep

- `rho in {0.05, 0.2}`
- Latent-only en iyi iki trial'dan başla
- Aynı split ve seed'leri kullan
- Value-aware cost ilk tura girmez

### 8.4 P3 kapısı

- Mixed action gap latent-only trial'dan kötü olmayacak.
- Action-oracle ve top-2 gap latent-only trial'a göre en az `%15` iyileşecek.
- Latent MSE `%20`den fazla bozulmayacak.
- Assignment'lar yalnızca actor'la ilgili dar bir command bin'ine çökmeyecek.

Bu kapı geçilmezse action-aware cost kullanılmaz; latent-only objective ile `P4` değerlendirilir.

---

## 9. P4: Additive on-policy minimal objective

Amaç, offline'da kanıtlanan objective'i mevcut mimari ve dünya üzerinde tek değişken olarak test etmektir.

### 9.1 Yeni task

Task adı:

`go2_moects_local`

Environment aynı `Go2MoECTS` sınıfını kullanır. Yalnız train config farklıdır.

Yeni config sınıfları:

```python
class Go2MoECTSLocalCfg(Go2MoECTSCfg):
    pass

class Go2MoECTSLocalCfgPPO(Go2MoECTSCfgPPO):
    class algorithm(Go2MoECTSCfgPPO.algorithm):
        local_loss_coef = ...
        router_loss_coef = ...
        responsibility_target_max = ...
        responsibility_epsilon = ...
        action_cost_coef = ...
        gate_prior_beta = 0.0

    class runner(Go2MoECTSCfgPPO.runner):
        experiment_name = "go2_moects_local"
        run_name = "moe_cts_local" + get_simulator_suffix()
```

### 9.2 Mevcut algorithm'e backward-compatible genişletme

`PPO_MOE_CTS.__init__` yeni opsiyonları `0`/disabled varsayılanlarla alacak:

```python
local_loss_coef=0.0
router_loss_coef=0.0
responsibility_epsilon=0.03
responsibility_tau=None
responsibility_target_max=None
action_cost_coef=0.0
gate_prior_beta=0.0
local_head_only=True
```

`go2_moects` bu alanları ya hiç set etmeyecek ya da açıkça sıfır kullanacak. Sıfır katsayılarda eski `_compute_encoder_losses()` numerik ve gradient eşdeğerliği testle pinlenecek.

Alternatif olarak yeni `PPO_MOE_CTS_Local` sınıfı yazılabilir; fakat encoder-update kodunun büyük bölümünü kopyalamak yerine ortak loss bundle hook'u tercih edilir:

```python
def _compute_student_encoder_objective(...) -> MoELossBundle:
    ...
```

Legacy ve local yollar aynı update loop'u kullanır.

### 9.3 Gerekli dosya değişiklikleri

| Dosya | Değişiklik |
|---|---|
| `rsl_rl/modules/moe_utils.py` | Component/head-only API |
| `rsl_rl/utils/moe_specialization.py` | Ortak objective ve metrikler |
| `rsl_rl/algorithms/ppo_moe_cts.py` | Opsiyonel local objective hook'u |
| `rsl_rl/runners/moe_cts_runner.py` | Yeni telemetry key'leri |
| `legged_gym/envs/go2/go2_moects/go2_moects_config.py` | Local config sınıfları |
| `legged_gym/envs/go2/go2_moects/__init__.py` | Yeni config export |
| `legged_gym/envs/__init__.py` | `go2_moects_local` registration |
| `legged_gym/utils/helpers.py` | Debug task set'ine local task |
| `legged_gym/scripts/play.py` | `moects_local` task type'ini MoE export/play yoluna ekle |

### 9.4 Yeni telemetry

TensorBoard key'leri:

```text
Loss/local_expert
Loss/router_kl
Loss/action_local
MoE/responsibility_entropy
MoE/responsibility_effective_experts
MoE/responsibility_max_weight
MoE/responsibility_usage_min
MoE/responsibility_usage_max
MoE/responsibility_usage_std
MoE/route_assignment_kl
MoE/top1_action_gap
MoE/top2_action_gap
MoE/action_oracle_gap
MoE/latent_oracle_gap
MoE/expert_individual_gap_mean
MoE/expert_individual_gap_min
MoE/expert_norm_ratio
MoE/route_switch_rate
```

Telemetry hesapları training'i belirgin yavaşlatmamalı. Pahalı expert-action matrisleri her iteration değil, config ile belirlenen aralıkta ilk student minibatch üzerinde hesaplanacak.

### 9.5 On-policy deney kolları

| Kol | Objective | Amaç |
|---|---|---|
| `B0` | Mevcut `go2_moects` | Tarihsel baseline |
| `B1` | Yeni kod, katsayılar `0` | Kod-path regression kontrolü |
| `L1` | `L_mix + L_local` | Expert sinyali tek başına |
| `L2` | `L_mix + L_local + L_route` | Önerilen minimal yol |
| `L3` | `L2 + action_cost` | Yalnız P3 geçtiyse |
| `N1` | Unweighted local negatif kontrol | Kopyalanma hipotezi |

### 9.6 Eğitim sırası

1. CPU/unit testler
2. 4-env iki-iteration Genesis smoke
3. 64-env 100-iteration wiring run
4. 1024-env 1000-iteration medium gate
5. 8192-env en az 3 seed 15k-iteration campaign
6. Kapı geçerse 30k tamamla

Smoke yalnız wiring, finite loss, checkpoint, resume ve export kanıtıdır; yöntem etkinliği kanıtı değildir.

### 9.7 P4 kapıları

#### 1000-iteration medium gate

- NaN/Inf yok
- Tüm expert'lar kullanılıyor
- Mixed action gap baseline'dan `%15`ten fazla kötü değil
- Top-2/action-oracle gap doğru yönde
- Student mean reward baseline bandının dışına çökmüyor
- Encoder grad norm clip'e sürekli yapışmıyor

#### 15k campaign gate

En az üç seed'de:

1. Median eval return baseline'dan en fazla `%3` düşük.
2. Fall rate baseline'dan mutlak `+2 puan`dan fazla kötü değil.
3. Tracking linear/yaw error baseline'dan `%5`ten fazla kötü değil.
4. Action-oracle ve top-2 gap en az iki seed'de baseline'dan `%30` iyi.
5. En az bir robustness ekseninde tutarlı kazanım var veya parameter-matched dense baseline'a karşı anlamlı avantaj var.
6. Assignment/gate ilişkisi en az bir **gözlenebilir** eksende seed'ler arası tekrar ediyor.

#### P4 öldürme kriterleri

- Seed'lerin yarısında expert collapse
- Specialization metrikleri artarken return/robustness faydası yok
- Dense baseline aynı sonucu daha az parametre/FLOP ile veriyor
- Route kimliği seed'ler arası optimal permutation sonrası bile kararsız
- Top-2 veya bireysel expert yetkinliği artmıyor

Bu durumda full residual mimari uygulanmaz.

---

## 10. P5: Parameter/FLOP-matched dense baseline

Semantic MoE geliştirmeden önce, mevcut kazanımın yalnızca kapasite kaynaklı olmadığı gösterilmelidir.

### 10.1 Baseline'lar

1. Mevcut `K=8` dense soft MoE
2. Global öğrenilebilir fakat input-bağımsız mixture weights
3. Uniform gate
4. `K=1` aynı backbone ailesi
5. Toplam parametre eşlenmiş dense history encoder
6. Aktif FLOP/latency eşlenmiş dense history encoder
7. Residual mimari aşamasında shared-base-only model

### 10.2 Eşleştirilecekler

- Trainable parametre sayısı
- Optimizer-state byte boyutu
- Forward FLOP
- Batch latency
- Tek-env deploy latency
- Peak GPU memory
- Teacher/actor/critic kapasitesi
- Latent boyutu ve normalizasyon
- Training sample/iteration bütçesi

### 10.3 MoE programını kapatma kriteri

Parameter-matched dense model:

- Median görev skorunda `%2` içinde,
- Action gap'te `%10` içinde,
- Robustness'ta anlamlı fark olmadan,
- Daha düşük latency/variance ile

aynı sonucu veriyorsa semantic MoE hattı durdurulur. Mevcut model alternatif bir parameterization olarak belgelenir.

---

## 11. P6: Full residual MoE mimarisi

Bu aşama yalnız `P4` ve `P5` kapıları geçilirse uygulanacaktır.

### 11.1 Yeni task ve sınıflar

Task:

`go2_moects_residual`

Yeni sınıflar:

```text
rsl_rl/modules/residual_moe_encoder.py
    ResidualMoEEncoder
    ResidualExpert

rsl_rl/modules/actor_critic_residual_moe_cts.py
    ActorCriticResidualMoECTS

rsl_rl/algorithms/ppo_residual_moe_cts.py
    PPO_RESIDUAL_MOE_CTS
```

Runner olarak mevcut `MoECTSRunner` kullanılabilir; algorithm aynı 5-tuple update sözleşmesini ve `moe_stats` yapısını korumalıdır.

### 11.2 Mimarinin ilk sürümü

Varsayılan `K=4`. Sekiz expert ilk sürümde kullanılmayacak.

```text
history: 225
  |
  v
shared stem: 225 -> 256 -> 128, ELU
  |----------------------|------------------------|
  v                      v                        v
base encoder          router                  residual experts
128 -> 64 -> 32       128 -> 64 -> K          K x (128 -> 64 -> 32)
  |                      |                        |
  b                  softmax(g)                 r_k
```

Expert candidate:

\[
z_{ik}=\operatorname{Norm}(b_i+\alpha r_{ik}).
\]

Mixed student latent:

\[
z_i=\operatorname{Norm}\left(b_i+\alpha\sum_kg_{ik}r_{ik}\right).
\]

Bu yapının amaçları:

- Ortak locomotion bilgisi `b` içinde kalır.
- Expert'lar sıfırdan tam latent öğrenmek yerine rejime özgü residual öğrenir.
- Yanlış route, base latent nedeniyle mevcut top-1 kadar yıkıcı olmaz.
- Full latent orthogonality yerine gerekirse residual diversity ölçülebilir.

### 11.3 Başlatma

- Base ve residual son katmanları ayrı init edilir.
- Residual final-layer gain başlangıçta `1e-2` mertebesinde küçük tutulur.
- Expert'lar tamamen aynı init edilmez; simetri küçük fakat deterministik biçimde kırılır.
- Router son bias'ı sıfır; başlangıç marginal'i uniform.
- `alpha=1` sabit olabilir; residual küçük init ile etkisi zaten kontrollüdür.
- İlk sürümde trainable alpha kullanılmaz; model alpha'yı sıfıra kaçırarak expert yolunu kapatmamalıdır.

### 11.4 Training schedule

#### `R0`: Base bootstrap

- Residual experts ve router frozen.
- Yalnız shared stem/base teacher latent'a distill edilir.
- Offline bankada early stopping.
- Hedef: base action gap mevcut dense student'a makul yakınlık.

#### `R1`: Residual specialization warm-up

- Stem/base frozen.
- Residual experts + router eğitilir.
- `L_mix + L_local + L_route`.
- Hard routing yok.
- Gate prior beta `0`.

#### `R2`: Joint fine-tune

- Stem/base açılır.
- Stem/base LR, residual/router LR'nin `0.1x`i.
- Local loss residual expert'lara gider.
- Mixed loss bütün student encoder'a gider.

#### `R3`: On-policy scratch campaign

Warm-start yalnız engineering pilot'tur. Bilimsel karşılaştırma için residual model scratch'ten, aynı seed/bütçe ile eğitilir.

### 11.5 Warm-start sınırı

Mevcut `model_23500` ile engineering pilot yapılırsa:

- Teacher encoder, actor, critic ve std ortak key'lerden yüklenebilir.
- Yeni student encoder offline bootstrap checkpoint'inden yüklenir.
- Main PPO ve student optimizer'lar fresh başlar.
- Bu koşu, baseline ile adil learning-curve karşılaştırması sayılmaz.
- Curriculum state restore edilirse manifestte açıkça işaretlenir.
- Kısmi `load_state_dict(strict=False)` sonucu missing/unexpected key listesi fail-closed kontrol edilir; sessiz key kaybı kabul edilmez.

### 11.6 Actor/critic ABI

- Teacher encoder değişmez.
- Actor ve critic mimarisi değişmez.
- Student `history_encoder.forward(history)` 32-D normalized latent döndürür.
- `act_student()` yine `[latent, obs]` sırasını kullanır.
- `evaluate(..., is_teacher=False)` student latent'ı detach ederek critic'e verir.
- PPO gradient izolasyonu mevcut contract testleriyle aynı kalır.

### 11.7 Export

`PolicyExporterMoECTS`, yeni encoder `forward()` ABI'si aynı kaldığı için tekrar kullanılabilir. Yine de ayrı test zorunludur:

- Python `act_student` ile JIT output eşleşmesi
- ONNX export smoke
- Batch `1` ve batch `N`
- CPU/GPU eşleşmesi
- Dense soft route modu
- Daha sonra eklenirse top-2 modu

`play.py` içinde `moects_residual` task type'ı MoE exporter ve history adapter yoluna eklenmelidir.

### 11.8 Checkpoint/resume

İlk residual sürümde stateful EMA veya adaptive usage controller olmayacak. Böylece checkpoint için:

- Model state
- Main PPO optimizer
- `history_encoder_optimizer_state_dict`
- Iteration
- Existing curriculum state

yeterlidir.

Stage, global iteration ve config stage boundaries'den deterministik olarak türetilir. Resume sonrası aynı stage'e dönüldüğü test edilir.

### 11.9 P6 ablation'ları

| Ablation | Amaç |
|---|---|
| Base only | Residual expert katkısı |
| `K=2,4,8` | Expert sayısı |
| Local loss kapalı | Mimarinin tek başına etkisi |
| Router KL kapalı | Assignment supervision etkisi |
| Shared final expert layer | Bağımsız residual head etkisi |
| Action cost kapalı/açık | Davranış-duyarlı cost |
| Dense soft vs top-2 mask | Sparsity distribution shift'i |
| Parametre-eşlenmiş dense | Kapasite confound'u |

### 11.10 P6 kabul kapısı

En az üç seed'de:

- Baseline return/fall/tracking zararı `P4` limitleri içinde.
- Parameter-matched dense baseline'a karşı en az bir robustness veya efficiency avantajı.
- Individual expert/top-2 competence `P4` minimal yoldan daha iyi.
- Route/expert ilişkisi en az bir gözlenebilir eksende tekrar edilebilir.
- Expert ablation hangi rejimin zarar gördüğünü tutarlı biçimde gösteriyor.
- Seed variance baseline'dan belirgin daha kötü değil.

Yalnız entropy/NMI artışı başarı sayılmaz.

---

## 12. P7: Opsiyonel uzantılar

Bu özellikler ilk residual implementasyona birlikte eklenmeyecektir.

### 12.1 Gate prior'lı responsibility

Yalnız assignment'lar gate'den bağımsız biçimde kararlıysa:

\[
q_k \propto g_k^\beta\exp(-C_k/\tau).
\]

`beta`, `0 -> 0.5` ramp edilir. Başlangıçtan `beta=1` kullanılmaz; mevcut gate tercihlerini pseudo-ground-truth'a dönüştürür.

### 12.2 Sinkhorn / balanced assignment

Uygulama ancak soft responsibility'de dead expert veya aşırı usage dengesizliği kalırsa.

- İlk olarak soft entropic Sinkhorn
- Minibatch değil büyük/replay batch
- Uniform column marginal zorunlu değil
- Gerçek rejim oranları için EMA veya bounded capacity prior
- Batch composition sensitivity testi

Kill kriteri: usage daha uniform olurken task/action metrikleri kötüleşiyorsa Sinkhorn kaldırılır.

### 12.3 Privileged training-only router

Önce ayrı observability deneyi:

1. Privileged oracle route accuracy/performance
2. 5-frame MLP student router
3. Daha uzun history
4. GRU/TCN router
5. Calibration ve posterior entropy

Student router privileged route'u uniform/majority baseline'dan anlamlı daha iyi tahmin edemiyorsa privileged routing uygulanmaz.

Terrain ID ilk hedef değildir. Öncelik:

- Command rejimi
- Contact/gait phase
- Slip/impact
- Recovery durumu
- Temas sonrası physics rejimi

### 12.4 EMA teacher

Yalnız teacher drift probe büyük hareket gösterirse:

- Aynı anchor bankada checkpoint'ler arası latent cosine
- Orthogonal Procrustes sonrası residual drift
- Actor action drift
- Student tracking lag

EMA kullanılırsa:

- `L_mix` güncel teacher'ı takip eder.
- Yalnız responsibility cost EMA teacher'dan gelebilir.
- EMA module actor-critic state dict içinde saklanır.
- Resume round-trip testi zorunludur.

### 12.5 Temporal consistency

Önce yalnız metrik:

- Gate total variation
- Argmax switch rate
- Contact/command değişimi olmadan switch rate
- Action jerk ile route switch korelasyonu

Gerçek chattering görülürse adjacent rollout pair'ları üzerinde temporal KL eklenir. Storage sırası doğrulanmadan shuffled minibatch satırları komşu kabul edilmez.

### 12.6 Top-2

Top-2 ancak dense training forward'ında da kullanılıp fine-tune edilecek. Dense-train/sparse-test yasak.

Başlatma kapısı:

\[
\text{top2 action gap} \le 1.2 \times \text{dense learned action gap}.
\]

İlk top-2 implementasyonu tüm residual expert'ları hesaplayıp mask uygulayabilir; bu yalnız davranış deneyidir. Gerçek FLOP kazanımı için sample'ları expert index'e göre gruplayan sparse dispatch ayrı bir deployment projesidir ve TorchScript/ONNX testi gerektirir.

---

## 13. Telemetry ve analiz paketi

### 13.1 Uzmanlaşma tek bir entropy metriği değildir

Rapor dört ayrı katman içermelidir:

1. **Distillation:** latent/action taklidi
2. **Routing:** entropy, usage, stability
3. **Expert competence:** tek expert/top-2/oracle/ablation
4. **Task impact:** return, fall, tracking, robustness, latency

### 13.2 Zorunlu expert analizleri

- Expert başına held-out action gap
- Her expert kapatıldığında delta return/action gap
- En iyi ve ikinci en iyi expert cost margin
- Expert specialization matrix: expert x regime
- Optimal permutation ile seed'ler arası expert hizalama
- Residual cosine/CKA
- Expert norm ve gradient norm
- Router input Jacobian/sensitivity özeti

### 13.3 Rejim eksenleri

- `vx` bin
- yaw bin
- Hızlanma/yavaşlama
- Contact bit pattern
- Stance/swing proxy
- Slip
- Impact/foot-force quantile
- Recovery/tilt
- Terrain family ve level
- Friction/mass/COM/control delay/PD gain

Terrain semantiği headline yapılmadan önce observability geçmelidir.

### 13.4 Task metrikleri

- Student ve teacher mean reward
- Eval return/return per step
- Fall rate/episode fall rate
- Tracking linear/yaw RMSE
- SPNTE
- Tilt
- Torque/power
- Action rate/smoothness
- Foot slip
- Recovery time
- DR-axis worst-case ve macro ortalama

### 13.5 Kanıt dili

- Bir checkpoint veya bir seed yöntem zaferi sayılmaz.
- Offline action gap, on-policy return yerine kullanılmaz.
- Gate entropy düşmesi tek başına uzmanlaşma sayılmaz.
- Terrain probe başarısızlığı "mutlak imkânsız" değil, "mevcut veri/probe ile gösterilmedi" diye raporlanır.
- Critical file/checkpoint hash'i doğrulanmadan deney tamamlandı denmez.

---

## 14. Test planı

### 14.1 Saf objective testleri

`tests/test_moe_specialization_utils.py`

- `q` satır toplamı `1`
- Tau azalınca responsibility entropy monoton azalır
- Epsilon smoothing minimum mass sağlar
- `q.detach()` nedeniyle assignment yoluna gradient yok
- Router KL expert parametrelerine gradient vermez
- Local head-only loss backbone'a gradient vermez
- Mixed loss backbone/heads/router'a gradient verir
- Teacher target gradient almaz
- Unweighted local loss simetrik expert kopyalama yönü verir
- Balance yalnız marginal usage'ı etkiler
- Action-aware functional actor parametrelerinde grad bırakmaz, latent'a grad verir

### 14.2 Legacy regression testleri

Mevcut `tests/test_moects_contract.py` genişletilecek:

- Yeni katsayılar `0` iken eski total loss eşit
- Aynı seed/input ile model parameter gradient'leri eşit
- Optimizer param domain'leri değişmemiş
- PPO surrogate student encoder'a ulaşmıyor
- Critic encoder'a ulaşmıyor
- Existing telemetry key'leri korunuyor

### 14.3 Component API testleri

- `forward_components` ile reconstruct edilen mixed latent `forward()` ile eşit
- `forward_with_weights` eski ABI'yi koruyor
- Gate satır toplamı `1`
- Expert output şekli `[B,K,D]`
- TorchScript `forward()` compile oluyor
- Eski checkpoint strict load oluyor

### 14.4 Offline CLI testleri

- Temp directory'de tiny synthetic bank
- Kaynak checkpoint overwrite edilmiyor
- Manifest/hash kontrolü
- Deterministik split
- Early stopping
- Best-val seçimi test split'e bakmıyor
- Atomic checkpoint/report write
- Resume offline optimizer round-trip
- V1 banka pilot, V2 banka confirmation etiketi

### 14.5 Task/runner/export testleri

Yeni veya genişletilecek testler:

```text
tests/test_moects_specialization_contract.py
tests/test_moects_residual_contract.py
tests/test_moects_export_paths.py
tests/test_moects_runner_contract.py
tests/test_moects_telemetry.py
```

Kontroller:

- `go2_moects_local` registry
- `go2_moects_residual` registry
- Class name `eval()` resolution in `cts_runner` namespace
- `MoECTSRunner` 5-tuple update contract
- Correct `RolloutStorageMoECTS`
- Interleaved role mapping
- JIT/ONNX export
- Debug env count `4`
- Auxiliary optimizer save/load
- Resume iteration/stage
- Legacy `go2_moects` config aynı

### 14.6 Genesis entegrasyon testleri

Simulator testleri seri çalıştırılacak:

```bash
env SIMULATOR=genesis WANDB_MODE=disabled \
  .venv/bin/python -m pytest tests/test_moects_specialization_contract.py -q

env SIMULATOR=genesis WANDB_MODE=disabled \
  .venv/bin/python tests/test_all_tasks.py \
  --tasks go2_moects_local --iterations 2 --headless

env SIMULATOR=genesis WANDB_MODE=disabled \
  .venv/bin/python tests/test_all_tasks.py \
  --tasks go2_moects_residual --iterations 2 --headless
```

Short smoke yalnız lifecycle ve wiring kanıtıdır.

---

## 15. Komut planı

### 15.1 V2 bank toplama

Planlanan CLI:

```bash
env SIMULATOR=genesis WANDB_MODE=disabled \
  .venv/bin/python legged_gym/scripts/eval/probe_moe_gate.py \
  --mode collect \
  --task go2_moects \
  --load_run Aug03_12-01-45_moe_cts_genesis \
  --ckpt 23500 \
  --num_envs 384 \
  --target_samples 100000 \
  --per_terrain_min 5000 \
  --sample_every 5 \
  --schema_version 2 \
  --seed 1
```

Seed `2` ayrı output dizinine toplanacak.

### 15.2 Offline pilot

```bash
env WANDB_MODE=disabled \
  .venv/bin/python -m legged_gym.scripts.eval.moe_specialization_offline \
  --config configs/eval/moe_specialization_offline.yaml \
  --run_dir logs/go2_moects/Aug03_12-01-45_moe_cts_genesis \
  --ckpt 23500 \
  --samples logs/eval/moe_gate_probe/Aug03_12-01-45_moe_cts_genesis/model_7500/samples.pt \
  --pilot-v1-bank
```

### 15.3 Offline confirmation

```bash
env WANDB_MODE=disabled \
  .venv/bin/python -m legged_gym.scripts.eval.moe_specialization_offline \
  --config configs/eval/moe_specialization_offline.yaml \
  --run_dir logs/go2_moects/Aug03_12-01-45_moe_cts_genesis \
  --ckpt 23500 \
  --samples <samples_v2.pt> \
  --split-manifest <split_manifest.json> \
  --confirmatory
```

### 15.4 Minimal on-policy

Laptop wiring:

```bash
env SIMULATOR=genesis WANDB_MODE=disabled \
  .venv/bin/python -m legged_gym.scripts.train \
  --task go2_moects_local \
  --headless --num_envs 64 --max_iterations 100
```

Büyük makine medium gate:

```bash
env SIMULATOR=genesis WANDB_MODE=disabled \
  .venv/bin/python -m legged_gym.scripts.train \
  --task go2_moects_local \
  --headless --num_envs 1024 --max_iterations 1000 --seed 1
```

Confirmatory koşu:

```bash
env SIMULATOR=genesis WANDB_MODE=disabled \
  .venv/bin/python -m legged_gym.scripts.train \
  --task go2_moects_local \
  --headless --num_envs 8192 --max_iterations 15000 --seed 1
```

Seed `2` ve `3` aynı config/commit ile ayrı koşulur.

---

## 16. Dosya ve sahiplik haritası

### 16.1 P0-P3

| Dosya | Sahip olduğu sözleşme |
|---|---|
| `legged_gym/scripts/eval/probe_moe_gate.py` | Veri toplama ve baseline ablation |
| `legged_gym/scripts/eval/probe_lib.py` | Ortak split/decoder yardımcıları |
| `legged_gym/scripts/eval/moe_specialization_offline.py` | Offline train/eval orchestration |
| `rsl_rl/utils/moe_specialization.py` | Saf objective ve metric matematiği |
| `rsl_rl/modules/moe_utils.py` | Expert component API |
| `configs/eval/moe_specialization_offline.yaml` | Deney konfigürasyonu |

### 16.2 P4

| Dosya | Sahip olduğu sözleşme |
|---|---|
| `rsl_rl/algorithms/ppo_moe_cts.py` | Legacy + opsiyonel local encoder objective |
| `rsl_rl/runners/moe_cts_runner.py` | Telemetry ve optimizer checkpoint |
| `go2_moects_config.py` | Local task config |
| `legged_gym/envs/__init__.py` | Task registration |
| `play.py` / `helpers.py` | Play/export/debug task classification |

### 16.3 P6

| Dosya | Sahip olduğu sözleşme |
|---|---|
| `rsl_rl/modules/residual_moe_encoder.py` | Shared base + residual experts |
| `rsl_rl/modules/actor_critic_residual_moe_cts.py` | Actor/critic/student integration |
| `rsl_rl/algorithms/ppo_residual_moe_cts.py` | Stage schedule ve specialization objective |
| `rsl_rl/modules/__init__.py` | Actor class export |
| `rsl_rl/algorithms/__init__.py` | Algorithm class export |
| `rsl_rl/runners/cts_runner.py` | `eval()` namespace import |
| `go2_moects_config.py` | Residual task config |
| `legged_gym/envs/__init__.py` | Residual task registration |

---

## 17. PR ve implementasyon sırası

### PR-1: Veri ve baseline

- Probe output overwrite düzeltmesi
- V2 schema/provenance
- Split manifest
- Baseline metrics ve hash manifesti
- Yalnız veri/test değişiklikleri

**Kapı:** P1 veri doğrulaması.

### PR-2: Offline objective

- Component API
- Saf specialization utility
- Offline fine-tune CLI
- Negative controls
- Unit ve offline integration testleri

**Kapı:** P2/P3 sonuç raporu. Bu rapor olmadan PR-3 başlamaz.

### PR-3: Minimal on-policy

- Backward-compatible algorithm hook'u
- `go2_moects_local`
- Telemetry
- Runner/export/debug contract
- Smoke ve medium gate

**Kapı:** 3-seed P4 kampanyası.

### PR-4: Dense baseline paketi

- Parameter/FLOP-matched dense encoder
- Latency/memory ölçümü
- Adil campaign config'i

**Kapı:** P5 MoE gereklilik kararı.

### PR-5: Residual mimari

- Yeni encoder/actor/algorithm
- `go2_moects_residual`
- Offline bootstrap
- Checkpoint/export/resume
- Full ablation paketi

**Kapı:** P6 3-seed sonuçları.

### PR-6+: Opsiyonel

Her özellik ayrı PR ve tek değişkenli ablation:

- Gate-prior responsibility
- Sinkhorn
- Privileged router
- EMA teacher
- Temporal consistency
- Top-2/sparse dispatch

---

## 18. Riskler ve azaltımlar

| Risk | Belirti | Azaltım |
|---|---|---|
| Mixed performans kaybı | Learned action gap artar | `L_mix` korunur, local gradient `%5-10` ile başlar |
| Rich-get-richer collapse | Bir expert usage'i baskın | Gate prior beta `0`, q smoothing, soft assignment |
| Expert kopyalanması | Functional cosine artar, individual gap aynı | Unweighted local kullanma; local responsibility |
| Backbone contamination | Tüm expert'lar birlikte kayar | Local branch'te features detach |
| Norm oyunu | Expert norm oranı büyür | Cost normalize expert üzerinde; norm telemetry/weight decay |
| Train/test leakage | Offline train iyi, test kötü | Episode/env/physics-cell holdout |
| Router observability | Privileged route tahmin edilemiyor | Ayrı observability kapısı; terrain id'yi zorlamama |
| Moving teacher | Assignment kimliği sürekli değişir | Önce drift probe; gerekirse q için EMA |
| Action-aware actor contamination | PPO actor gradyanı kirlenir | Detached functional actor call |
| Resume farkı | Loss/stage restart sonrası sıçrar | Optimizer + iteration + config fingerprint testleri |
| Export mismatch | JIT action Python'dan farklı | Latent-first exporter equality testi |
| Kapasite confound'u | MoE daha büyük olduğu için iyi | Parameter/FLOP-matched dense baseline |
| Metric gaming | Entropy iyileşir, policy kötüleşir | Task kapıları entropy'den önce gelir |
| Route chattering | Action jerk/fall artar | Önce metrik, gerekirse temporal KL |

---

## 19. Tamamlanma tanımı

### Minimal kurtarma tamamlanmış sayılır, eğer:

- P2 confirmation iki collection seed'inde geçer.
- P4 en az üç training seed'inde baseline performansını korur.
- Bireysel/top-2 expert yetkinliği önceden tanımlı eşik kadar artar.
- En az bir gözlenebilir rejim ekseninde tekrar edilebilir specialization vardır.
- Parameter-matched dense baseline'a karşı task/robustness/efficiency gerekçesi vardır.
- Resume ve export contract'ları geçer.

### Full residual yöntem tamamlanmış sayılır, eğer:

- P6 üç seed'de minimal yola veya dense baseline'a karar-değiştirici avantaj sağlar.
- Expert ablation gerçek ve tekrar edilebilir rejim etkisi gösterir.
- Kazanım yalnız entropy/NMI değil, return/robustness/efficiency metriğine yansır.
- Export edilen policy aynı davranışı verir.
- Checkpoint resume aynı stage/objective ile devam eder.

### Program durdurulur, eğer:

- Soft local objective bireysel expert yetkinliği oluşturmaz.
- Yetkinlik artışı task performansına veya robustness'a yansımaz.
- Dense baseline aynı sonucu daha basit verir.
- Student history hedef route'u gözleyemez.
- Yöntem seed'ler arası kararsızdır.

Bu durumda mevcut `go2_moects`, semantic uzmanlar olarak değil, yararlı bir dense conditional basis encoder olarak korunur ve doğru isimlendirmeyle raporlanır.

---

## 20. İlk uygulanacak somut iş paketi

Bu belgeden sonra doğrudan full mimariye geçilmeyecek. İlk iş paketi yalnız şunlardır:

1. `probe_moe_gate.py` için output-dir ve V2 provenance schema.
2. `rsl_rl/utils/moe_specialization.py` saf loss/metric fonksiyonları.
3. `moe_utils.py` backward-compatible component API.
4. Frozen `moe_specialization_offline.py` script'i.
5. P0/P1/P2 unit ve integration testleri.
6. Mevcut v1 bankada pilot.
7. V2 bankada confirmation.
8. Sonuç raporu ve açık `PASS / PARTIAL / STOP` kararı.

`go2_moects_local` kodu yalnız bu karar `PASS` veya action-aware sonrası gerekçeli `PARTIAL` olursa başlatılacaktır.

---

## 21. Literatür dayanağı

- Jacobs, Jordan, Nowlan ve Hinton, **Adaptive Mixtures of Local Experts**, 1991: local expert competition ve gating.
- Jordan ve Jacobs, **Hierarchical Mixtures of Experts and the EM Algorithm**, 1994: soft responsibilities ve EM ayrımı.
- Lewis ve ark., **BASE Layers**, ICML 2021: balanced assignment kapasite/yük problemidir; semantik garanti değildir.
- Zhou ve ark., **Expert Choice Routing**, NeurIPS 2022: expert-side capacity control.
- Puigcerver ve ark., **From Sparse to Soft Mixtures of Experts**, ICLR 2024: dense differentiable mixture'lar hard semantic routing olmadan faydalı olabilir.
- Dai ve ark., **StableMoE**, ACL 2022: routing fluctuation ve distillation/freeze.
- Yang ve ark., **Multi-expert Learning of Adaptive Legged Locomotion**, Science Robotics 2020: robotikte expert kimliği güçlü skill pretraining ile kurulmuştur; mevcut CTS yolunda böyle bir prior yoktur.

Literatürün ortak sonucu: load balance tek başına uzmanlaşma değildir; local responsibility rekabet yaratır fakat semantiği veri, gözlenebilirlik ve maliyet tanımı belirler.

## 22. Upstream MuJoCo karar kapısı (2026-08-07)

Tam kanıt ve route tabloları: [upstream_mujoco_moe_uzmanlasma_sonuclari.md](upstream_mujoco_moe_uzmanlasma_sonuclari.md). 137k/164k bridge’leri diffuse gate + farklı functional expert basis ile uyumlu; terrain-semantic routing ve method win kanıtlanmadı. Bu nedenle sonraki kapı, yeni mimariye geçmeden önce üç-seed frozen-bank/20 s ayırt edici deneyidir; entropy veya action MSE tek başına `PASS` değildir.
