# Sade ve Ayrıştırıcı Eval V2 Planı

## 1. Amaç ve temel sözleşme

Mevcut Faz A/B artifact’leri değiştirilmeyecek; tarihsel sonuç olarak arşivlenecek. Yeni değerlendirme ayrı bir `eval_v2` kampanyası olacak ve üç açık katmandan oluşacak:

1. **In-distribution:** Eğitim dağılımında genel tracking, düşme riski ve return.
2. **Tek-eksen sweep:** Her seferinde yalnız bir fizik parametresi değişecek; diğer fizik parametreleri nominal kalacak.
3. **Zorlayıcı senaryolar:** Artan bumpy-terrain, lateral push ve command step/reversal testleri.

Her katmanda yalnız aynı ana metrik sözleşmesi kullanılacak:

- `tracking`
  - Forward/lateral komutlarda lineer hız hatası: `||v_cmd_xy - v_base_xy||`.
  - Yaw komutunda açısal hız hatası: `|yaw_rate_cmd - yaw_rate_base|`.
  - ID tablosunda lineer ve açısal tracking ayrı gösterilecek.
- `fall_rate`
  - Sabit ölçüm penceresinde en az bir kez erken düşen environment yüzdesi.
  - Episode sayısına bölünmeyecek; ID, sweep ve zor senaryolarda aynı anlama gelecek.
- `return_per_step`
  - Ölçüm penceresindeki tüm reward’ların environment-step başına ortalaması.
  - Tamamlanmamış episode’lar atılmayacak.
  - Checkpoint seçmekte kullanılmayacak; destek metriği olacak.

Eski `mean_return`, headroom kapıları, çok sayıda enerji/slip/settling metriği ve Faz A/B ayrımı yeni ana rapora taşınmayacak. Gereken tanısal ham sinyaller raw artifact’te tutulabilir fakat varsayılan raporda görünmez.

## 2. Checkpoint seçimi

### Mevcut dokuz run

Altay’da dokuz run’ın her birinde `model_0.pt`–`model_3000.pt` arasında 16 periyodik checkpoint olduğu canlı olarak doğrulandı. Yeniden eğitim yapılmadan offline seçim uygulanacak:

1. Her checkpoint aynı deterministic validation scenario bank’inde değerlendirilecek.
2. Validation bank:
   - `768` environment.
   - Ayrı validation seed’i: `31001`.
   - Flat terrain.
   - Friction, mass ve CoM eğitim ID dağılımında.
   - Eğitimdeki push dağılımı açık.
   - Command alanı `vx, vy, yaw ∈ [-1,1]`.
   - Command örnekleri rastgele çağrı sırasına bırakılmayacak; önceden üretilmiş stratified scenario bank kullanılacak.
   - `warmup=100`, `measured_steps=1100`.
3. Tracking seçim skoru:
   ```text
   tracking_score =
       0.5 × (
           tracking_lin_err / sqrt(2)
           + tracking_yaw_err
       )
   ```
   Normalizasyon, command sınırlarının lineer eksenlerde `[-1,1]²`, yaw’da `[-1,1]` olmasına dayanacak.
4. Leksikografik seçim:
   - `fall_rate ≤ 0.05` olan checkpoint’ler “güvenli” kabul edilecek.
   - En az bir güvenli checkpoint varsa yalnız güvenli kümede en düşük `tracking_score` seçilecek.
   - Hiç güvenli checkpoint yoksa önce en düşük `fall_rate`, eşitlikte en düşük `tracking_score` seçilecek.
   - Tam eşitlikte daha erken iterasyon seçilecek.
5. Seçim sonucunda ağırlık kopyalanmayacak ve run klasörleri değiştirilmeyecek. `checkpoint_selection.json`, her `model × training_seed` için seçilen dosya yolu, iterasyon, SHA-256, tracking, fall ve validation protokolünü kaydedecek.
6. Final eval, validation’dan bağımsız `eval_seed=41001` scenario bank’ini kullanacak.

### Gelecek eğitimler

- In-training eval aynı tracking+fall kuralına geçirilecek.
- Yeni checkpoint adı `best_tracking.pt` olacak.
- `--ckpt best_tracking` açıkça desteklenilecek.
- Geriye uyumluluk için `--ckpt best`, önce `best_tracking.pt`, yoksa legacy `best.pt` çözecek.
- Checkpoint metadata’sında selection metric, validation seed, fall threshold ve seçildiği iterasyon bulunacak.
- Training commit’i temiz olmalı; manifest `git_commit`, `git_dirty` ve dirty ise `git_diff_sha256` taşımalı.

## 3. Genel campaign engine

İlk kampanya [go2_tripler_v2.yaml](/home/emin/code/online-estimation/genesis-wp/LeggedGym-Ex/configs/eval/go2_tripler_v2.yaml) olacak; runner ve rapor kodu üç yönteme hard-code edilmeyecek.

Önerilen ana arayüz:

```bash
python -m legged_gym.scripts.eval.campaign plan \
  --config configs/eval/go2_tripler_v2.yaml

python -m legged_gym.scripts.eval.campaign select-checkpoints \
  --config configs/eval/go2_tripler_v2.yaml \
  --shard 0/1

python -m legged_gym.scripts.eval.campaign run \
  --config configs/eval/go2_tripler_v2.yaml \
  --suite id|axes|hard|all \
  --shard 0/1 \
  --resume

python -m legged_gym.scripts.eval.campaign aggregate \
  --config configs/eval/go2_tripler_v2.yaml

python -m legged_gym.scripts.eval.campaign report \
  --config configs/eval/go2_tripler_v2.yaml
```

Campaign YAML şunları tanımlayacak:

- Kampanya adı ve artifact root.
- Simulator ve eval commit beklentisi.
- Model etiketi, task adı ve training-seed → run-folder eşlemesi.
- Checkpoint-selection manifest’i.
- Validation/final eval seed’leri.
- Warmup, measured steps ve replica sayıları.
- Command case’leri.
- Axis registry, grid, nominal değer, ID aralığı ve birimi.
- Hard-scenario seviyeleri.
- Rapor renkleri ve yöntem sırası.

Runner davranışı:

- `plan` yalnız hücreleri ve tahmini process/env sayısını listeler.
- `--resume`, metadata ve gerekli anahtarları geçerli artifact’leri atlar.
- Çıktılar `.partial.npz` olarak yazılıp doğrulamadan sonra atomik biçimde nihai ada taşınır.
- Task/run/checkpoint/seed/hash uyuşmazlıkları fail-loud olur.
- Aynı GPU’da aynı anda yalnız bir Genesis eval prosesi çalışır.
- Birden fazla GPU kullanılırsa deterministic `--shard i/n` ile hücreler bölünür; her GPU yine seri çalışır.
- Login node’da eval çalıştırılmaz; yalnız kısa inventory/Slurm/transfer işlemleri yapılır.
- Campaign engine ilk sürümde LeggedGym-Ex + Genesis sözleşmesini destekler; uyumlu yeni task/model eklemek YAML değişikliğiyle mümkün olur. Desteklenmeyen simulator veya axis açık hatayla reddedilir.

## 4. Katman 1 — In-distribution eval

Her seçilmiş `model × training_seed` checkpoint’i için:

- `num_envs=768`
- `warmup=100`
- `steps=2000`
- `eval_seed=41001`
- Flat terrain.
- Observation noise açık.
- Friction, mass, CoM ve training push dağılımları açık.
- Command’ler stratified scenario bank’ten gelir:
  - `vx, vy, yaw ∈ [-1,1]`
  - Alanın uçları ve sıfıra yakın bölgeler zorunlu olarak kapsanır.
  - Command schedule 10 saniyede bir önceden belirlenmiş yeni komuta geçer.
  - Aynı scenario ID, bütün yöntemlerde aynı fizik/command/push başlangıcını temsil eder.

ID rapor tablosu:

| Model | Tracking linear | Tracking yaw | Fall rate | Return/step |
|---|---:|---:|---:|---:|

Her hücre training seed’ler üzerinden:

```text
median [min, max], n=3
```

olarak gösterilecek. `256/768 env = seed` gibi bir ifade hiçbir yerde kullanılmayacak; bilimsel tekrar birimi training seed olarak kalacak.

## 5. Katman 2 — Tek-eksen sweep

### Ortak protokol

Her sweep’te:

- Flat terrain.
- Training push kapalı.
- Yalnız seçilen axis değişir.
- Diğer bütün axis’ler nominale sabitlenir.
- Observation noise açık.
- `warmup=100`, `steps=2000`.
- Her `axis_value × command_magnitude` hücresinde `64` replica.
- Aynı axis ve hareket yönündeki dört command büyüklüğü tek Genesis prosesinde environment bloklarına paketlenir.
- Command modları:
  - Forward: `(vx,vy,yaw)=(m,0,0)`
  - Lateral: `(0,m,0)`
  - Yaw: `(0,0,m)`
- Büyüklükler:
  ```text
  m ∈ {0.5, 1.0, 1.5, 2.0}
  ```
- `0.5` ve `1.0` command-ID, `1.5` ve `2.0` command-OOD olarak metadata ve raporda açıkça etiketlenir.
- Negatif yönler axis sweep’e eklenmez; full ID bank negatif command’leri kapsar, step-response ise `+/-` reversal içerir.

### Çekirdek yedi axis

| Axis | Grid | Training-ID |
|---|---|---|
| `added_mass` | `[-2,-1,0,1,2,3,4,5] kg` | `[-1,1]` |
| `friction` | `[0.1,0.25,0.4,0.5,0.7,0.9,1.1,1.25,1.5,1.75,2.0,2.5]` | `[0.5,1.25]` |
| `com_x` | `[-0.08,-0.05,-0.03,0,0.03,0.05,0.08] m` | `[-0.03,0.03]` |
| `com_y` | aynı | `[-0.03,0.03]` |
| `com_z` | aynı | `[-0.03,0.03]` |
| `pd_gain_scale` | `[0.5,0.65,0.8,1.0,1.2,1.35,1.5]` | yalnız nominal `1.0` |
| `control_delay` | `[0,1,2,3,4,5,6]` control step | yalnız nominal `0` |

Control period `20 ms` olduğu için delay raporunda hem step hem milisaniye gösterilecek:

```text
0, 20, 40, 60, 80, 100, 120 ms
```

### Axis uygulama sözleşmesi

Axis registry her axis için şu arayüze sahip olacak:

- `name`, `unit`, `grid`, `nominal`, `in_distribution`.
- `prepare_cfg(env_cfg)` — build öncesi gereken simulator yapısını hazırlar.
- `apply(env, per_env_values)` — fizik ve varsa privileged-label buffer’ını birlikte yazar.
- `pin_nominal(env)` — non-swept axis’i nominale getirir.
- `validate(env)` — simulator’ın uyguladığı gerçek değeri geri okuyup grid ile karşılaştırır.

Özel durumlar:

- P5 axis setter’ları hem gerçek fiziği hem oracle label buffer’ını güncelleyecek.
- PD gain setter, aynı scale’i tüm joint’lerin `kp` ve `kd` değerlerine uygulayacak; reset sırasında yeniden randomize edilmeyecek.
- Control-delay build sırasında action queue’yu oluşturacak. Eval-fixed modunda reset queue’yu temizleyecek fakat delay değerini yeniden çekmeyecek.
- Her artifact, istenen grid ile simulator’dan geri okunan gerçek değerleri taşıyacak; uyuşmazlıkta çalışma başarısız olacak.

## 6. Katman 3 — Zorlayıcı senaryolar

Zor senaryolar severity curve üretecek; tek bir “geçti/kaldı” noktası olmayacak.

### Bumpy terrain

- Deterministik aynı temel height-map deseni kullanılacak; yalnız genlik ölçeklenecek.
- Maksimum mutlak yükseklik:
  ```text
  0, 2.5, 5, 7.5, 10 cm
  ```
- Held command: `vx=1.0`, diğerleri `0`.
- Training push ve diğer fizik randomization’ları nominale sabitlenecek.
- Her seviye `256` environment, `warmup=100`, `steps=2000`.
- Heightfield hash’i metadata’ya yazılacak; bütün yöntemler aynı deseni görecek.

### Deterministik lateral push

Bu test “force impulse” diye adlandırılmayacak; simulator doğrudan base lateral velocity değişimi uyguladığı için raporda **lateral velocity-kick** olarak geçecek.

- Held command: `vx=1.0`.
- Kick zamanı: warmup sonrası `50.` measured step.
- `dvy`:
  ```text
  0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0 m/s
  ```
- Recovery window: `250` step.
- Tüm seviyeler tamamlanacak; yüksek fall görüldü diye adaptif erken durdurma yapılmayacak.
- Ana tracking metriği recovery penceresindeki lineer tracking hatası olacak.
- Aynı üç ana çıktı: recovery tracking, pencere fall rate’i, return/step.

### Command step/reversal

İki ayrı şiddette çalışacak:

```text
amplitude = 1.0  # command-ID sınırı
amplitude = 2.0  # command-OOD stres
```

Schedule:

```text
stand
→ forward(+A)
→ reverse(-A)
→ lateral(+A)
→ yaw(+A)
→ stop
```

- Her faz `150` step.
- Nominal fizik, flat terrain, push kapalı.
- Forward/reverse/lateral fazlarında lineer tracking; yaw fazında açısal tracking kullanılacak.
- Rapor faz başına tracking, fall ve return/step gösterecek.
- Settling time, peak tilt ve benzeri değerler raw artifact’te tutulabilir; ana tabloda gösterilmeyecek.

## 7. Veri modeli ve artifact düzeni

Yeni kampanya eski `benchmark_tripler_2026-07-13` ağacının üzerine yazmayacak:

```text
logs/eval/go2_tripler_v2/
├── campaign.yaml
├── manifest.json
├── checkpoint_selection.json
├── commands.log
├── ledger.csv
├── raw/
│   ├── validation/
│   ├── id/
│   ├── axes/
│   └── hard/
├── tables/
│   ├── results.csv
│   └── summary.csv
└── report/
    └── index.html
```

### Raw NPZ

Raw artifact yalnız özet ortalamaları değil, yeniden aggregate edilebilmesi için per-environment değerleri taşıyacak:

- Scenario/cell kimliği.
- Model/task/training seed.
- Run folder, checkpoint iterasyonu/yolu/hash’i.
- Training ve eval commit/provenance.
- Eval/scenario seed.
- Axis, gerçek axis değerleri ve ID/OOD maskesi.
- Command mode/magnitude/matrisi ve command-ID/OOD maskesi.
- Per-env tracking, ever-fell ve reward-per-step.
- Warmup, steps, replica sayısı.
- Simulator/GPU/Python/Torch/Genesis sürümleri.
- Terrain veya scenario-bank hash’i.

### `results.csv`

Tidy long format kullanılacak; her satır bir:

```text
model × training_seed × scenario_cell × metric
```

olacak. Temel kolonlar:

```text
campaign, suite, model, task, training_seed,
checkpoint_iter, checkpoint_sha256,
scenario, axis, axis_value, axis_is_id,
command_mode, command_magnitude, command_is_id,
severity, metric, mean, p25, p50, p75,
num_replicas, eval_seed
```

### `summary.csv`

Training seed’ler üzerinden:

```text
value_median, value_min, value_max, n_training_seeds
```

üretilecek.

Yöntem karşılaştırması karmaşık gate yerine tutarlılık sayısı taşıyacak:

- Tracking/fall için daha düşük olan lehine.
- Return/step için daha yüksek olan lehine.
- Pairwise:
  - `P5 better than MLP: k/3`
  - `P5+V better than P5: k/3`
- Yüzde headroom varsayılan özet olmayacak.
- Üç seed ile confidence interval veya p-değeri üretilmeyecek.

## 8. Tek ve estetik HTML rapor

Ayrı Faz A/Faz B HTML’leri yerine tek rapor üretilecek.

### Üst özet

- Kampanya/checkpoint/provenance bilgisi.
- `n=3 training seed` uyarısı.
- ID tablosu.
- Her suite’in completeness durumu.
- Validation ve final scenario bank’lerinin farklı olduğu bilgisi.

### Axis explorer

Tek filtrelenebilir panel:

- Axis seçici.
- Movement: forward/lateral/yaw.
- Speed: `0.5/1.0/1.5/2.0`.
- Metric: tracking/fall/return-per-step.
- Yöntem renkleri sabit.
- Çizgi: training-seed medyanı.
- Bant: seed min–max.
- İsteğe bağlı “seed çizgilerini göster” anahtarı.
- Physics-ID bölgesi gri gölge.
- `1.5/2.0` seçildiğinde command-OOD rozeti.
- Yanında ham değer ve `k/3` tutarlılık tablosu.

Yaw görünümünde mutlaka açısal tracking kullanılacak; mevcut Faz B’deki lineer-yaw grafik hatası tekrarlanmayacak.

### Hard-scenario görünümü

Üç sekme:

- Terrain severity.
- Velocity-kick severity.
- Step/reversal.

Her sekmede yalnız tracking, fall ve return/step bulunacak. HTML dış CDN kullanmayacak; grafik verisi gömülü JSON ve küçük yerel JavaScript/SVG renderer ile çizilecek. Rapor, yüzlerce base64 PNG taşıyan büyük bir dosyaya dönüşmeyecek.

## 9. Doğrulama ve kabul testleri

### Birim testleri

- Tracking-best seçim kuralı:
  - Güvenli checkpoint her zaman güvensizi geçer.
  - Güvenli kümede tracking belirleyicidir.
  - Hiç güvenli yoksa fall, sonra tracking belirleyicidir.
- `reward_per_step`, tamamlanmamış episode reward’larını kaybetmez.
- `fall_rate`, pencere içinde en az bir düşüşü doğru sayar.
- Forward/lateral tracking lineer, yaw tracking açısaldır.
- Campaign YAML eksik/yanlış model, run, grid veya checkpoint’te fail-loud olur.
- Scenario bank aynı ID için bütün gözlem boyutlarında aynı fizik/command/push değerlerini üretir.
- Yedi axis’in nominal pinning ve gerçek-değer readback testleri.
- PD gain ve control delay değerleri environment resetinden sonra değişmez.
- Control delay `step ↔ ms` dönüşümü doğrudur.
- Command packing, dört magnitude’ı doğru environment bloklarına yerleştirir.
- Aggregator metric yönüne göre `k/3` tutarlılığını doğru hesaplar.

### GPU smoke testleri

- Üç task × bir checkpoint.
- Küçük ID bank.
- Yedi axis’in her biri için iki grid noktası ve iki command magnitude.
- Bir terrain, bir kick ve iki-fazlı kısa step test.
- Observation boyutları `45/50/53`.
- Policy/checkpoint/task eşleşmesi.
- NaN/Inf kontrolü.
- Artifact atomic promotion ve `--resume`.
- Aynı checkpoint/scenario iki kez çalıştırıldığında tracking farkı önceden belirlenen toleransın altında olmalı:
  ```text
  absolute difference ≤ 1e-3
  ```

### Tam kampanya kabulü

- Dokuz mevcut run için 144 checkpoint-validation artifact’i tamamlanmış.
- Dokuz `best_tracking` seçimi manifestte kayıtlı.
- ID: 9 artifact.
- Axis: `7 axis × 3 movement × 9 model-seed = 189` artifact; her artifact dört command magnitude ve tüm axis grid’ini içerir.
- Hard:
  - Terrain: `5 severity × 9 = 45` artifact.
  - Push: 9 artifact; bütün kick seviyeleri aynı artifact’te.
  - Step: 9 artifact; iki amplitude aynı artifact’te.
- Manifest bütün beklenen hücreleri tamamlanmış göstermeden final rapor “complete” olarak işaretlenmez.
- `results.csv`, `summary.csv` ve HTML aynı raw artifact hash setinden üretilir.

## 10. Sabit varsayımlar

- İlk V2 kampanyası mevcut MLP, P5 ve P5+V modellerinin training seed `1,2,3` run’larını kullanacak.
- Seed `4–5` eğitilmeyecek.
- Üç seed nedeniyle sonuçlar betimsel olacak; “kanıtlandı” dili kullanılmayacak.
- Mevcut dirty-worktree training provenance’i raporda uyarı olarak gösterilecek; ağırlıklar geçersiz sayılmayacak.
- Mevcut Faz A/B verileri yeni ana rapora karıştırılmayacak; yalnız tasarım gerekçesi ve regresyon kontrolü olarak saklanacak.
- Faz B’de görülen P5+V step-response ayrışması ve P5 friction-yaw ayrışması yeni protokolün zorlayıcı senaryoları korumasına gerekçe oldu; bunların aynı yönde çıkması kabul testi olmayacak.
- Tüm yeni kod ve campaign tanımı temiz bir eval commit’inde dondurulmadan tam veri üretimine başlanmayacak.
