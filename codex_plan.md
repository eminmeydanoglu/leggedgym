# P5 Privileged Policy Headroom Doğrulama Planı

## Özet

[oracle_adam_etme.md](/home/emin/code/online-estimation/genesis-wp/LeggedGym-Ex/oracle_adam_etme.md) mevcut kısa taslak korunmadan, doğrudan ayrıntılı ve uygulanabilir bir deney planına dönüştürülecek.

Bu fazın tek bilimsel sorusu:

> Aynı P5 domain-randomization dağılımında ve aynı performanstan bağımsız command schedule altında, gerçek P5 bilgisini gören politika standart DR-MLP’ye tekrarlanabilir bir performans avantajı sağlıyor mu?

İlk kampanya yalnızca dar eşleşmiş çifti kapsayacak:

- `go2_bench_mlp`: 45 boyutlu proprioception, P5 erişimi yok.
- `go2_bench_oracle_id`: aynı 45 boyutlu proprioception + gerçek P5.
- İlk training seed’leri: `1, 2, 3`.
- Sonuç karışıksa eklenecek seed’ler: `4, 5`.
- `RMA`, `DreamWaQ`, `SysID`, wide ve rich politikalar bu karar kapısı kapanana kadar eğitilmeyecek.

## 1. Bilimsel sözleşme ve adlandırma

- İnsan tarafından okunan bütün açıklamalarda “oracle ceiling” ve “mutlak üst sınır” ifadeleri kaldırılacak.
- Tercih edilen adlandırma:
  - Genel: **P5 privileged policy**
  - Dar görev: **narrow P5 privileged policy**
  - Kavramsal kısaltma gerektiğinde: **P5 oracle**
- Kayıtlı task adları ve eski checkpoint yolları bozulmayacak; `go2_bench_oracle_id` gibi teknik isimler değiştirilmeyecek.
- P5 açık biçimde şu sırayla tanımlanacak:
  `P5 = [friction, added_base_mass, com_x, com_y, com_z]`.
- Push, observation noise ve command değerleri P5’e dahil edilmeyecek:
  - Push geçici dış bozucudur.
  - Command zaten actor observation içindedir.
  - P5 yalnızca episode boyunca sabit kalan gizli fizik parametrelerini temsil eder.
- “Oracle headroom” şu anlama gelecek:
  > Aynı eğitim ve değerlendirme dağılımında, gerçek P5 erişiminin DR-MLP’ye göre sağladığı ölçülmüş avantaj.
- Teorik oracle ile öğrenilmiş PPO politikası ayrılacak; P5 bilgisinin mevcut olması, PPO’nun onu iyi kullanacağını garanti etmez.

## 2. Protokol düzeltmeleri

### Ortak command schedule

Mevcut performansa bağlı curriculum kapatılacak. Bütün benchmark yöntemleri için ortak, iterasyon-temelli schedule kullanılacak:

| Training iterasyonu | `lin_vel_x` dağılımı |
|---|---|
| `0–499` | `[-0.5, 0.5]` |
| `500–2999` | `[-1.0, 1.0]` |

- `lin_vel_y=[-1,1]`, `ang_vel_yaw=[-1,1]`, heading, zero-command olasılığı ve command resampling süresi değişmeyecek.
- Schedule politika başarısına, reward’a veya episode uzunluğuna bağlı olmayacak.
- Iterasyon `500` başında yeni aralık bütün environment’lara uygulanacak ve mevcut command’lar yeniden örneklenecek.
- Resume edilen bir run, checkpoint iterasyonuna göre doğru schedule aşamasından devam edecek.
- Schedule, aynı per-step command dizisini garanti etmek için değil, bütün yöntemlerin aynı command dağılımını aynı training aşamasında görmesini garanti etmek için kullanılacak.

Bunun için opt-in runner arayüzü eklenecek:

```python
runner.command_schedule = [
    {"start_iteration": 0, "lin_vel_x": [-0.5, 0.5]},
    {"start_iteration": 500, "lin_vel_x": [-1.0, 1.0]},
]
```

Schedule yalnızca bu alanı tanımlayan benchmark config’lerinde etkin olacak; diğer repo task’larının davranışı değişmeyecek.

### Training seed ve run kimliği

- Eğitim CLI’ına `--seed <int>` eklenecek.
- Bu değer `train_cfg.seed` ve ardından `env_cfg.seed` için tek kaynak olacak.
- Benchmark run klasörleri seed’i açıkça taşıyacak:
  - `..._bench_mlp_genesis_seed1`
  - `..._bench_oracle_id_genesis_seed1`
- Her run klasörüne şu alanları içeren `run_manifest.json` yazılacak:
  - task
  - training seed
  - git commit
  - simulator ve sürümleri
  - P5 dağılımı
  - command schedule
  - max iteration
  - checkpoint-selection protokolü

### Checkpoint seçimi

- Config’teki tam validation command alanı `lin_vel_x=[-1,1]` olacak.
- `best.pt`, curriculum’dan bağımsız sabit validation alanında seçilecek.
- Validation protokolü:
  - Sabit eval seed: `12345`
  - Tam command alanı
  - In-distribution P5
  - Deterministik inference
  - Aynı `steps`, `warmup` ve fall guard
- `best.pt` seçim skoru mevcut `mean_return + fall guard` yaklaşımını koruyacak; headroom kararı ise test verisindeki `tracking_lin_err` üzerinden verilecek.
- Checkpoint içine `training_seed`, `eval_iteration`, `eval_score` ve schedule aşaması kaydedilecek.
- Nihai OOD/headroom testleri checkpoint seçmek için kullanılmayacak.

## 3. Eğitim ve değerlendirme kampanyası

### Eğitim

İlk aşamada toplam altı run çalıştırılacak:

```text
go2_bench_mlp       × seed 1, 2, 3
go2_bench_oracle_id × seed 1, 2, 3
```

Her run:

- `3000` PPO iterasyonu
- Aynı environment sayısı
- Aynı PPO bütçesi ve network hidden dimensions
- Aynı P5 eğitim bandı
- Aynı reward, push, noise ve command schedule
- Yalnız actor observation boyutunda beklenen fark: `45` ve `50`

Komut şablonu:

```bash
env SIMULATOR=genesis WANDB_MODE=disabled \
  .venv/bin/python -m legged_gym.scripts.train \
  --task <go2_bench_mlp|go2_bench_oracle_id> \
  --headless --seed <1|2|3> --max_iterations 3000
```

### Birincil headroom test hücreleri

Karar kapısı yalnızca önceden belirlenen in-distribution hücrelerden hesaplanacak:

- `added_mass ∈ {-1, 0, +1} kg`
- `command_vx ∈ {0.75, 1.0} m/s`
- `command_vy=0`
- `command_yaw=0`

Böylece her politika/seed için altı birincil hücre olacak. Her hücre:

- `256` paralel environment
- `100` warmup step
- `2000` ölçüm step’i
- Sabit eval seed `12345`
- Diğer P5 bileşenleri nominal
- Deterministik policy

Birincil metrik:

```text
tracking_lin_err — düşük daha iyi
```

Güvenlik metriği:

```text
fall_rate — düşük daha iyi
```

Destek metrikleri:

- `mean_ep_len`
- `falls_per_1k`
- `tracking_ang_err`
- `mean_return`
- `torque_sq`
- `mech_power`
- `foot_slip`

### İkincil açıklayıcı testler

Bunlar karar eşiğini değiştirmeyecek; avantajın nerede ortaya çıktığını açıklayacak:

1. **Tam mass sweep**
   - `−2, −1, 0, 1, 2, 3, 4, 5 kg`
   - `vx=0.5` ve `vx=1.0`
   - In-distribution ve OOD davranışı ayrı raporlanacak.

2. **Friction excitation**
   - Tam friction grid’i.
   - Üç ayrı command koşulu:
     - İleri: `(vx, vy, yaw)=(1.0, 0, 0)`
     - Lateral: `(0, 0.75, 0)`
     - Yaw: `(0, 0, 0.75)`
   - Amaç, önceki yavaş düz-yürüyüş friction testinin ölü eksen olmasını gidermek.

3. **CoM sweep**
   - `com_x`, `com_y`, `com_z`
   - Her seferinde yalnız bir bileşen değişecek.
   - Diğer P5 bileşenleri nominal olacak.
   - Ana command `vx=1.0`.

4. **Command step-response**
   - Fazlar:
     `stand → forward(1.0) → reverse(-1.0) → lateral(0.75) → yaw(0.75) → stop`
   - Her faz `150` step.
   - Settling time, error integral ve peak error raporlanacak.

5. **Push recovery**
   - `command_vx=1.0`
   - `dvy=1.0`
   - Nominal fizik ve in-distribution `added_mass=+1 kg`
   - `+5 kg` yalnız OOD diagnostic olarak ayrıca çalıştırılabilir; headroom kapısına dahil edilmeyecek.

### Artifact düzeni

Sonuçlar mevcut Wave‑1 çıktılarının üzerine yazılmayacak:

```text
/home/emin/code/online-estimation/logs/eval/wave1_recheck/
├── manifest.json
├── seed_1/
├── seed_2/
├── seed_3/
├── aggregate/
└── go2_wave1_recheck_report.html
```

Her `.npz` dosyası şunları taşıyacak:

- task ve training seed
- eval seed
- run klasörü ve checkpoint
- git commit
- fizik ekseni/grid’i
- command koşulu
- steps, warmup ve environment replica sayısı

## 4. Headroom karar kapısı

Her training seed `s` ve birincil test hücresi `c` için göreli headroom:

```text
h(s,c) = 100 ×
         (tracking_err_MLP(s,c) − tracking_err_P5(s,c))
         / tracking_err_MLP(s,c)
```

Her seed’in özeti:

```text
H(s) = altı birincil hücredeki h(s,c) değerlerinin medyanı
```

### Üç-seed kararı

**Geç:**

- `H(1)`, `H(2)` ve `H(3)` değerlerinin üçü de pozitif.
- Üç seed’in medyan headroom’u en az `%10`.
- Full in-distribution değerlendirmede P5 politikasının fall rate’i hiçbir seed’de MLP’den `0.02` mutlak orandan fazla kötü değil.

**Erken başarısız:**

- Üç seed’in tamamında `H(s) ≤ 0`.

**Seed 4–5’e genişlet:**

- Yukarıdaki iki sonuçtan hiçbiri oluşmazsa.
- Headroom aynı yönde fakat `%10` eşiğinin altındaysa.
- Bir seed güvenlik guard’ını ihlal ediyorsa.

### Beş-seed nihai kararı

**Geç:**

- En az `4/5` seed’de `H(s)>0`.
- Beş seed’in medyan headroom’u en az `%10`.
- En az `4/5` seed güvenlik guard’ını sağlıyor.
- Hiçbir seed’de P5 fall-rate farkı `+0.05` değerini aşmıyor.

Bunlardan biri sağlanmazsa headroom kapısı **başarısız** kabul edilecek.

### Sonraki adım

- Kapı geçerse:
  - İlk yöntem minimal `history → P5` estimator olacak.
  - Aynı P5 tanımı ve aynı critic fairness protokolü korunacak.
  - Estimator’ın DR-MLP ile P5 privileged policy arasına yerleşip yerleşmediği test edilecek.
  - RMA/DreamWaQ ancak minimal estimator sonucu görüldükten sonra açılacak.

- Kapı başarısız olursa:
  - P5 estimator/RMA/DreamWaQ kampanyası başlatılmayacak.
  - Ana bulgu “mevcut düz-zemin P5 benchmarkı adaptation-demanding değil; geniş DR yeterli” olarak raporlanacak.
  - Rich-P, PD gain ve control delay ayrı bir yeni hipotez/faz olarak tasarlanacak; mevcut P5 sonucuyla karıştırılmayacak.

## 5. Testler ve kabul ölçütleri

### Statik ve birim testleri

- `--seed` CLI override’ı hem train hem env config’e ulaşmalı.
- Schedule helper’ı şu sınırları doğru döndürmeli:
  - iterasyon `0` → `[-0.5,0.5]`
  - iterasyon `499` → `[-0.5,0.5]`
  - iterasyon `500` → `[-1,1]`
  - resume iterasyonu `≥500` → `[-1,1]`
- `go2_bench_mlp` ve `go2_bench_oracle_id` config diff’i yalnız gözlem/P5 ve run adıyla sınırlı olmalı.
- P5 sırası ve actor observation boyutları doğrulanmalı:
  - MLP: `45`
  - P5 privileged policy: `50`
- In-distribution evaluator’ın `lin_vel_x=[-1,1]` kullandığı test edilmeli.
- Sweep sırasında yalnız seçilen eksenin değiştiği, diğer P5 bileşenlerinin nominal kaldığı doğrulanmalı.

### Runtime smoke testleri

- Her iki task `16–64` environment ve kısa iteration sayısıyla başlatılmalı.
- Training log’unda seed ve command schedule aşaması görünmeli.
- Schedule sınırı küçültülmüş test config’iyle geçiş anı doğrulanmalı.
- `best.pt` oluşturulmalı ve checkpoint metadata’sı okunmalı.
- Aynı task/seed checkpoint’i `indist.py`, `sweep.py` ve `transient.py` tarafından yüklenebilmeli.

### Rapor kabul ölçütleri

- Grafiklerde her training seed ayrı nokta/çizgi olarak görünmeli.
- `256 environment`, `256 bağımsız training seed` şeklinde raporlanmamalı; istatistiksel tekrar birimi training seed olmalı.
- Dar eğitim bandı ve OOD alanları doğru gölgelenmeli.
- Birincil gate hücreleri ile ikincil diagnostic sonuçlar görsel ve metinsel olarak ayrılmalı.
- Rapor, sonuç ne olursa olsun önceden tanımlanmış karar kuralını değiştirmemeli.

## Varsayımlar

- Mevcut `oracle_adam_etme.md` doğrudan genişletilecek; ek shadow dosyası oluşturulmayacak.
- İlk kampanya yalnız `go2_bench_mlp ↔ go2_bench_oracle_id` dar çiftidir.
- Seed planı önce `1,2,3`, yalnız karışık sonuçta `4,5` şeklindedir.
- P5, DR bandı, reward, flat terrain, push ve observation noise değiştirilmez.
- Wide, rich, RMA, DreamWaQ ve SysID bu fazın kapsamı dışındadır.
- Eski task adları, checkpoint’ler ve Wave‑1 artifact’leri geriye dönük uyumluluk için korunur.
