# GO2 Üçlü Benchmark Planı

## 1. Amaç

Bu kampanya aynı eğitim protokolüyle eğitilmiş üç politika ailesini eşleşmiş training seed’ler üzerinden karşılaştıracak:

| Kısa ad | Task | Actor girdisi | Boyut | Critic girdisi | Boyut |
|---|---|---:|---:|---|---:|
| `MLP` | `go2_bench_mlp` | gürültülü proprioception | 45 | temiz proprioception + gerçek `base_lin_vel` | 48 |
| `P5` | `go2_bench_oracle_id` | gürültülü proprioception + gerçek P5 | 50 | temiz proprioception + gerçek P5 + gerçek `base_lin_vel` | 53 |
| `P5+V` | `go2_bench_oracle_id_vel` | gürültülü proprioception + gerçek P5 + gerçek `base_lin_vel` | 53 | temiz proprioception + gerçek P5 + gerçek `base_lin_vel` | 53 |

P5 sırası değişmeyecek:

```text
P5 = [friction, added_base_mass, com_x, com_y, com_z]
```

Kampanya üç soruyu birbirinden ayıracak:

1. **P5 değeri:** `MLP ↔ P5`
   - Gerçek P5 bilgisi dar, eşleşmiş domain-randomization bandında tekrarlanabilir avantaj sağlıyor mu?
2. **Velocity değeri:** `P5 ↔ P5+V`
   - P5 zaten bilinirken gerçek base linear velocity bilgisini actora vermek ek avantaj sağlıyor mu?
   - Bu en temiz actor-input karşılaştırmasıdır; iki yöntemin critic girdisi aynıdır.
3. **Birleşik ayrıcalık:** `MLP ↔ P5+V`
   - P5 ve velocity bilgisinin birleşik erişimi ne kadar toplam avantaj sağlıyor?
   - Bu karşılaştırma tanısaldır; tek başına “P5 etkisi” veya “velocity etkisi” diye yorumlanmayacaktır.

Önemli nedensellik sınırı: `MLP ↔ P5` karşılaştırmasında yalnız actor değil, critic’in P5 erişimi de değişmektedir. Bu nedenle ölçülen sonuç saf bir actor-input müdahalesi değil, mevcut **P5 ayrıcalıklı eğitim koşulunun toplam headroom’u** olarak adlandırılacaktır.

## 2. Doğrulanmış başlangıç durumu

Altay’da 13 Temmuz 2026 tarihinde doğrulanan eğitim commit’i:

```text
d138a2de490d0daef28841e848c74584aa51f147
```

Dokuz run’ın tamamında hem `best.pt` hem `model_3000.pt` vardır:

| Yöntem | Seed | Run klasörü |
|---|---:|---|
| MLP | 1 | `Jul13_09-47-39_bench_mlp_genesis_seed1` |
| MLP | 2 | `Jul13_09-48-03_bench_mlp_genesis_seed2` |
| MLP | 3 | `Jul13_09-48-28_bench_mlp_genesis_seed3` |
| P5 | 1 | `Jul13_09-48-26_bench_oracle_id_genesis_seed1` |
| P5 | 2 | `Jul13_09-48-51_bench_oracle_id_genesis_seed2` |
| P5 | 3 | `Jul13_09-49-17_bench_oracle_id_genesis_seed3` |
| P5+V | 1 | `Jul13_10-34-00_bench_oracle_id_vel_genesis_seed1` |
| P5+V | 2 | `Jul13_10-34-29_bench_oracle_id_vel_genesis_seed2` |
| P5+V | 3 | `Jul13_10-35-00_bench_oracle_id_vel_genesis_seed3` |

Her `run_manifest.json`, doğru task’ı, training seed’i ve `d138a2d` commit’ini taşımaktadır.

Uzak eğitim checkout’u ile mevcut yerel çalışma ağacı aynı değildir:

- Altay eğitim checkout’u: `d138a2d`
- Yerel `HEAD`: `39d0c67`
- Yerel çalışma ağacında commitlenmemiş benchmark değişiklikleri vardır.

Bu nedenle bütün yerel repo Altay’a körlemesine `rsync` edilmeyecek. Eval düzeltmeleri dar bir commit/patch olarak hazırlanıp eğitim commit’i üzerinde uygulanacak; checkpoint’ler ve run klasörleri değiştirilmeyecek.

## 3. Değişmeyecek deney sözleşmesi

- Training seed’leri eşleşmiş üçlülerdir: `1, 2, 3`.
- Eval seed bütün doğrulayıcı koşullarda `12345` olacaktır.
- Deterministik inference kullanılacaktır.
- Observation noise mevcut benchmark sözleşmesindeki gibi açık kalacaktır.
- Terrain, reward, action space, control ayarları, P5 training bandı ve PPO bütçesi değiştirilmeyecektir.
- Dar in-distribution P5 bandı:
  - `friction ∈ [0.5, 1.25]`
  - `added_mass ∈ [-1, 1] kg`
  - `com_x, com_y, com_z ∈ [-0.03, 0.03] m`
- Tam command alanı:
  - `lin_vel_x ∈ [-1, 1]`
  - `lin_vel_y ∈ [-1, 1]`
  - `ang_vel_yaw ∈ [-1, 1]`
- `256` paralel environment, `256 training seed` değildir. Bilimsel tekrar birimi training seed’dir; environment replikaları bir seed içindeki rollout dağılımını daha iyi ölçer.
- OOD sonuçları checkpoint seçmek için kullanılmayacaktır.
- `mean_return` yalnız destek metriğidir. Ana sonuç `tracking_lin_err`; güvenlik guard’ı `fall_rate` olacaktır.

## 4. Faz 0 — Eval harness ve provenance hazırlığı

Gerçek benchmark başlamadan önce aşağıdaki dar düzeltmeler tamamlanıp test edilecek.

### 4.1 `best.pt` yükleme desteği

Mevcut `sweep.py`, `indist.py` ve `transient.py` içindeki `--ckpt` argümanı yalnız tamsayı kabul ediyor; `get_load_path()` ise yalnız `model_<iter>.pt` seçiyor. Bu haliyle `best.pt` çalıştırılamaz.

Zorunlu düzeltme:

- `--ckpt best`, `--ckpt latest` ve açık iterasyon (`--ckpt 3000`) desteklenmeli.
- Seçilen gerçek dosya yolu ekrana ve `.npz` metadata’sına yazılmalı.
- `best.pt` içinden kaydedilmiş `iter`, `training_seed`, `eval_iteration`, `eval_score` ve schedule stage okunup artifact’e aktarılmalı.
- Checkpoint SHA-256 değeri kaydedilmeli.
- Yanlış task/run/checkpoint eşleşmesi fail-loud olmalı.

### 4.2 Step-response düzeltmesi

Mevcut kod `--step_yaw` argümanını tanımlıyor fakat schedule’a yaw fazı eklemiyor. Ayrıca yardım metnindeki `|vx|<=0.5` ifadesi eğitimin son `[-1,1]` command alanına göre eskidir.

Doğrulanacak yeni schedule:

```text
stand → forward(1.0) → reverse(-1.0) → lateral(0.75) → yaw(0.75) → stop
```

- Her faz `150` step.
- Yaw fazı gerçekten `env.commands[:, 2]` alanına yazılmalı.
- Faz isimleri, sınırları ve gerçek command matrisi `.npz` içine kaydedilmeli.
- Yaw fazı için angular error integral, peak angular error ve angular settling time da kaydedilmeli; yalnız linear-error metriğiyle yaw performansı yorumlanmamalı.
- Birim test schedule’da altı faz ve doğru değerleri doğrulamalı.

### 4.3 Artifact metadata’sı

Her çıktı en az şunları taşımalı:

- task ve yöntem etiketi
- training seed ve eval seed
- run klasörü
- checkpoint türü (`best`/`3000`), gerçek checkpoint iterasyonu ve SHA-256
- training commit ve eval commit
- simulator, Genesis, Python, Torch ve NumPy sürümleri
- GPU modeli
- axis, grid ve in-distribution bandı
- command koşulu
- `per_point`, warmup ve measured steps
- tarih/saat ve hostname

`indist.py` çıktısı da sweep ile aynı provenance seviyesine çıkarılmalı.

### 4.4 Otomasyon ve kabul testleri

Tek bir idempotent orchestration komutu hazırlanmalı. Bu komut:

- aşağıdaki sabit run map’ini kullanmalı;
- var olan geçerli `.npz` dosyasını yeniden çalıştırmamalı;
- eksik/bozuk çıktıyı fail-loud göstermeli;
- her alt komutu ve exit code’u `commands.log`/`ledger.tsv` içine yazmalı;
- yalnız bir GPU eval sürecini aynı anda çalıştırmalı;
- ara sonuçları atomik adla yazıp başarıdan sonra nihai ada taşımalı.

Benchmark öncesi smoke test:

```text
3 task × 1 checkpoint × küçük env/step
```

Kabul ölçütleri:

- Gözlem boyutları sırasıyla `45`, `50`, `53`.
- Her task kendi checkpoint’ini yükler.
- `--ckpt best` gerçekten `best.pt` yükler.
- `.npz` içindeki task, training seed, checkpoint hash’i ve run manifest’i uyuşur.
- Seçilen eksen dışında bütün P5 bileşenleri nominaldir.
- Aynı eval komutu yeniden çağrıldığında mevcut artifact korunur.

## 5. Checkpoint protokolü

### Birincil checkpoint

Ana tablolar ve karar kapıları `best.pt` ile üretilecektir. `best.pt`, eğitim sırasında sabit in-distribution validation alanında `mean_return + fall guard` protokolüyle seçildiği için OOD verisine bakılarak seçilmiş değildir.

### Duyarlılık checkpoint’i

Altı birincil hücre `model_3000.pt` ile de tekrarlanacaktır. Amaç checkpoint seçiminin sonucu değiştirip değiştirmediğini görmek, iki checkpoint arasından iyi görüneni sonradan seçmek değildir.

- `best.pt` ve `model_3000.pt` aynı karar yönünü verirse sonuç checkpoint-robust kabul edilir.
- İşaret veya kapı kararı değişirse bulgu **checkpoint-sensitive** olarak etiketlenir; güçlü headroom iddiası kurulmaz.
- İki checkpoint’in sonuçları birleştirilip örnek sayısı artırılmış gibi kullanılmaz.

## 6. Faz A — Zorunlu doğrulayıcı benchmark

Faz A tamamlanmadan geniş OOD ve transient sonuçlarından bilimsel sonuç çıkarılmayacaktır.

### A1. Full in-distribution değerlendirme

Her `yöntem × training_seed` için:

- checkpoint: `best.pt`
- `num_envs=256`
- `warmup=100`
- `steps=2000`
- eval seed: `12345`
- tam command alanı ve training P5 dağılımı

Toplam: `3 yöntem × 3 seed = 9` eval.

Amaç:

- Genel gait kalitesini ve safety guard’ını ölçmek.
- `tracking_lin_err`, `tracking_ang_err`, `fall_rate`, `mean_ep_len`, `falls_per_1k` raporlamak.
- `mean_return`, effort ve slip metriklerini yalnız destekleyici olarak tutmak.

Komut şablonu:

```bash
env SIMULATOR=genesis WANDB_MODE=disabled \
  .venv/bin/python -m legged_gym.scripts.eval.indist \
  --task <TASK> --load_run <RUN> --ckpt best \
  --num_envs 256 --warmup 100 --steps 2000 --seed 12345 \
  --out <OUT>/best/seed_<S>/indist/<TASK>.npz
```

### A2. Altı birincil headroom hücresi

Her `yöntem × training_seed` için:

- `added_mass ∈ {-1, 0, +1} kg`
- `command_vx ∈ {0.75, 1.0} m/s`
- `command_vy=0`
- `command_yaw=0`
- diğer P5 bileşenleri nominal
- `per_point=256`
- `warmup=100`
- `steps=2000`
- eval seed: `12345`

Bir sweep çağrısı üç mass noktasını birlikte çalıştırır. Toplam:

```text
3 yöntem × 3 training seed × 2 vx = 18 sweep
```

Komut şablonu:

```bash
env SIMULATOR=genesis WANDB_MODE=disabled \
  .venv/bin/python -m legged_gym.scripts.eval.sweep \
  --task <TASK> --load_run <RUN> --ckpt best \
  --axis added_mass --grid -1 0 1 \
  --command_vx <0.75|1.0> --command_vy 0 --command_yaw 0 \
  --per_point 256 --warmup 100 --steps 2000 --seed 12345 \
  --out <OUT>/best/seed_<S>/primary/<TASK>_mass_vx<VX>.npz
```

### A3. `model_3000.pt` duyarlılığı

A2’nin aynı `18` sweep’i yalnız `--ckpt 3000` ve ayrı artifact ağacıyla tekrarlanır. Full in-distribution koşulu yalnız `best.pt` safety sonucunda anormallik veya checkpoint uyuşmazlığı görülürse `model_3000.pt` için de çalıştırılır.

## 7. Headroom tanımları ve karar kapıları

Her training seed `s` ve birincil hücre `c` için, hücredeki `256` environment’ın `tracking_lin_err` ortalaması kullanılacaktır.

### 7.1 P5 headroom

```text
h_P5(s,c) = 100 × (err_MLP(s,c) - err_P5(s,c)) / err_MLP(s,c)
H_P5(s)   = altı h_P5(s,c) değerinin medyanı
```

### 7.2 Velocity headroom

```text
h_V(s,c) = 100 × (err_P5(s,c) - err_P5+V(s,c)) / err_P5(s,c)
H_V(s)   = altı h_V(s,c) değerinin medyanı
```

### 7.3 Birleşik tanısal headroom

```text
h_total(s,c) = 100 × (err_MLP(s,c) - err_P5+V(s,c)) / err_MLP(s,c)
H_total(s)   = altı h_total(s,c) değerinin medyanı
```

`H_total` ayrı bir karar kapısı değildir; iki ayrıcalığın toplam etkisini ve olası etkileşimi açıklamak için kullanılır.

### 7.4 Üç-seed kapısı

P5 ve velocity karşılaştırmaları için kapı ayrı ayrı uygulanır.

**Geç:**

- Üç `H(s)` değerinin tamamı pozitiftir.
- Üç seed’in medyan `H(s)` değeri en az `%10`’dur.
- Full in-distribution testinde avantajlı yöntemin fall rate’i hiçbir seed’de referanstan `0.02` mutlak orandan fazla kötü değildir.
- `model_3000.pt` duyarlılığı ana yönü tersine çevirmemektedir.

**Erken başarısız:**

- Üç seed’in tamamında `H(s) ≤ 0`.

**Seed 4–5’e genişlet:**

- Yukarıdaki iki sonuçtan hiçbiri oluşmazsa;
- yön aynı fakat `%10` eşiği geçilmezse;
- bir seed safety guard’ını ihlal ederse;
- `best` ve `model_3000` sonucu anlamlı biçimde çelişirse.

Genişletme gerekirse deney tasarımı korunacak ve üç yöntemin tamamı seed `4, 5` ile eğitilecektir; yalnız iyi/kötü görünen bir çift seçilmeyecektir.

### 7.5 Sonuçların yorumu

| P5 kapısı | Velocity kapısı | Yorum |
|---|---|---|
| Geç | Geçmez | Dar P5 bilgisi yararlı; doğrudan velocity’nin ek katkısı gösterilmedi. İlk estimator hedefi P5 olabilir. |
| Geç | Geç | Hem statik P5 hem velocity bilgi değeri taşır. Estimator hedefi `P5 + base_lin_vel` olmalıdır. |
| Geçmez | Geç | Kazanç P5’ten çok doğrudan velocity erişimiyle ilişkili. Sonuç P5 estimator’ını gerekçelendirmez. |
| Geçmez | Geçmez | Mevcut düz-zemin görevinde bu ayrıcalıklar için tekrarlanabilir headroom gösterilmedi. |
| Karışık | Karışık | Önceden belirlenen şekilde seed `4, 5` üçlüleri eklenir; diagnostic sonuçlarla kapı değiştirilmez. |

## 8. Faz B — Açıklayıcı benchmarklar

Faz B sonuçları A fazındaki kapıyı değiştirmeyecek; farkın nerede ve neden oluştuğunu açıklayacaktır. Birincil artifact’ler dondurulup `summary_primary.json` üretildikten sonra çalıştırılacaktır.

### B1. Tam added-mass sweep

- Grid: `{-2, -1, 0, 1, 2, 3, 4, 5} kg`
- Commands: `vx=0.5` ve `vx=1.0`
- Diğer P5 nominal
- `best.pt`, `per_point=256`, `warmup=100`, `steps=2000`
- In-distribution `[-1,1]` bölgesi grafiklerde gölgelenecek; `>1` ve `<-1` OOD olarak etiketlenecek.

### B2. Friction excitation

- Varsayılan tam friction grid’i:
  `{0.1, 0.25, 0.4, 0.5, 0.7, 0.9, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5}`
- Üç ayrı command:
  - ileri: `(vx, vy, yaw)=(1.0, 0, 0)`
  - lateral: `(0, 0.75, 0)`
  - yaw: `(0, 0, 0.75)`
- `best.pt`, `per_point=256`, `warmup=100`, `steps=2000`

Bu üç command aynı dosyada karıştırılmayacak; her biri ayrı artifact olacaktır.

### B3. CoM sweep

- Eksenler: `com_x`, `com_y`, `com_z`
- Grid: `{-0.08, -0.05, -0.03, 0, 0.03, 0.05, 0.08} m`
- Command: `(vx, vy, yaw)=(1.0, 0, 0)`
- Her seferinde yalnız bir CoM bileşeni değişecek; diğer P5 bileşenleri nominal kalacak.
- `best.pt`, `per_point=256`, `warmup=100`, `steps=2000`

### B4. Command step-response

- Schedule:
  `stand → forward(1.0) → reverse(-1.0) → lateral(0.75) → yaw(0.75) → stop`
- Her faz `150` step.
- Nominal fizik.
- `best.pt`, `per_point=256`, `warmup=100`, eval seed `12345`.
- Ana metrikler:
  - faz bazlı settling time
  - error integral
  - peak linear/angular tracking error
  - peak tilt
  - fall rate

### B5. Deterministik push recovery

Zorunlu iki fizik noktası:

1. nominal fizik
2. in-distribution `added_mass=+1 kg`

Koşullar:

- held command: `(vx, vy, yaw)=(1.0, 0, 0)`
- aynı sabit impulse: `dvy=1.0 m/s`, `dvx=0`
- `pre_push_steps=50`
- `recovery_steps=150`
- `best.pt`, `per_point=256`, `warmup=100`, eval seed `12345`

`added_mass=+5 kg` yalnız OOD diagnostic olarak sonradan çalıştırılabilir; karar kapısına girmez.

Ana metrikler:

- recovery error integral
- recovery time
- peak error ve peak tilt
- recovery-window fall rate

## 9. İstatistik ve raporlama

- Her training seed ayrı satır/nokta/çizgi olarak gösterilecek.
- Ana özet, training seed’ler üzerindeki medyan olacaktır.
- Yalnız üç seed varken dar bir confidence interval “kesin kanıt” gibi sunulmayacak.
- `256` environment replikası bağımsız training repeat gibi sayılmayacak ve p-değerini yapay biçimde küçültmek için kullanılmayacak.
- Aynı training seed numaraları yöntemler arasında eşleşmiş karşılaştırma olarak korunacak.
- Cell-level yüzde farklarının yanında mutlak `tracking_lin_err` değerleri de verilecek; küçük payda kaynaklı şişmiş yüzde farkları gizlenmeyecek.
- Fall guard mutlak oran farkıyla raporlanacak.
- Effort/slip/return metrikleri çoklu ikincil metriklerdir; bunlar üzerinden sonradan yeni başarı ölçütü seçilmeyecek.
- In-distribution ve OOD noktaları tablolarda ve grafiklerde açıkça ayrılacak.
- `best.pt` ve `model_3000.pt` grafikleri üst üste yığılmayacak; ayrı sensitivity panelinde gösterilecek.

## 10. Artifact düzeni

Yeni sonuçlar eski Wave-1 veya recheck çıktılarının üzerine yazılmayacaktır:

```text
logs/eval/benchmark_tripler_2026-07-13/
├── manifest.json
├── run_map.tsv
├── ledger.tsv
├── commands.log
├── environment.txt
├── best/
│   ├── seed_1/
│   │   ├── indist/
│   │   ├── primary/
│   │   ├── sweeps/
│   │   └── transient/
│   ├── seed_2/
│   └── seed_3/
├── model_3000/
│   ├── seed_1/primary/
│   ├── seed_2/primary/
│   └── seed_3/primary/
└── aggregate/
    ├── primary_cells.csv
    ├── seed_headroom.csv
    ├── safety.csv
    ├── checkpoint_sensitivity.csv
    ├── summary_primary.json
    ├── plots/
    └── benchmark_tripler_report.html
```

`manifest.json` tamamlanmış hücreleri, beklenen dosya listesini, training/eval commit’lerini ve protokol hash’ini taşıyacaktır. Rapor yalnız manifest’in bütün zorunlu Faz A hücrelerini tamamlanmış göstermesi halinde nihai olarak üretilecektir.

## 11. Altay A100 yürütme akışı

Login node yalnız kısa Slurm ve metadata işlemleri için kullanılacak. Eval süreçleri compute node’da çalışacaktır.

### 11.1 İnteraktif A100 alma

Aktif `interactive` işi yoksa login node üzerinden sleeper job gönderilecek:

```bash
ssh btutak@altay.uhem.itu.edu.tr 'bash -s' <<'EOF'
set -euo pipefail
mkdir -p /ari/users/btutak/auv/tmp /ari/users/btutak/auv/logs/slurm
cat > /ari/users/btutak/auv/tmp/interactive_a100_sleep.sbatch <<'SCRIPT'
#!/usr/bin/env bash
#SBATCH -J interactive
#SBATCH -A kds3by
#SBATCH -p a100q
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --time=04:00:00
#SBATCH --chdir=/ari/users/btutak/auv
#SBATCH --output=/ari/users/btutak/auv/logs/slurm/%x-%j.out
#SBATCH --error=/ari/users/btutak/auv/logs/slurm/%x-%j.err
sleep 4h
SCRIPT
sbatch /ari/users/btutak/auv/tmp/interactive_a100_sleep.sbatch
EOF
```

Durum:

```bash
ssh btutak@altay.uhem.itu.edu.tr "squeue -u btutak -o '%i %j %t %M %D %R'"
```

Job `R` olduğunda:

```bash
ssh makine
hostname
nvidia-smi -L
~/auv/bin/auv-env
cd /ari/users/btutak/auv/sim/genesis-wp/LeggedGym-Ex
source /ari/users/btutak/auv/sim/genesis-wp/activate.sh
```

`ssh makine '<komut>'` kullanılmayacak; `makine` alias’ı remote command kabul etmez. Uzun eval, server, watcher veya GPU işi login node’da çalıştırılmayacaktır.

### 11.2 Çalıştırma sırası

1. Compute node/GPU/environment doğrulaması.
2. Run map, checkpoint metadata ve SHA-256 preflight’i.
3. Üç task için küçük smoke test.
4. Faz A1: dokuz in-distribution eval.
5. Faz A2: on sekiz `best.pt` primary sweep.
6. Primary completeness ve metadata doğrulaması.
7. Faz A3: on sekiz `model_3000.pt` sensitivity sweep.
8. `summary_primary.json`, headroom tablosu ve karar kapısı.
9. Artifact’leri yerel makineye `rsync -a` ile yedekleme.
10. Faz B’yi yeni/uygun interaktif allocation’da sırayla çalıştırma.
11. Nihai HTML raporu ve plot doğrulaması.

GPU’da birden fazla Genesis eval aynı anda başlatılmayacaktır. Her proses bittikten sonra `.npz` anahtarları, shape’lar, NaN/Inf durumu ve metadata doğrulanmadan sıradaki hücreye geçilmeyecektir.

## 12. Faz A sonunda verilecek karar

İlk interaktif A100 oturumunun bilimsel çıktısı yalnız şu dört maddeden oluşmalıdır:

1. Dokuz `best.pt` için full in-distribution güvenlik ve tracking tablosu.
2. Her seed için `H_P5`, `H_V` ve tanısal `H_total`.
3. `best.pt ↔ model_3000.pt` checkpoint duyarlılığı.
4. Önceden tanımlanan sonuçlardan biri:
   - P5 kapısı geçti/geçmedi/karışık,
   - velocity kapısı geçti/geçmedi/karışık,
   - seed `4,5` gerekli/gereksiz.

Faz B sonuçları bu kararın eşiğini veya birincil metriğini geriye dönük değiştirmeyecektir.



aldığın interaktif node'un ismi interaktic olsun. 