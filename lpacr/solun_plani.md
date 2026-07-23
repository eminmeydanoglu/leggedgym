# V5 / UED — LP-ACRL ile Otomatik Episode-Distribution Design

## 0. Karar ve tek cümlelik hikâye

Bu çalışma, online estimation yöntemlerini yeniden karşılaştıran bir benchmark değildir. V4'ün Genesis/Go2 locomotion substrate'ını sabit tutup **episode'ların hangi command–terrain task'larından örnekleneceğini** değiştiren bağımsız bir Unsupervised Episode Design (UED) çalışmasıdır.

Ana sonuç cümlesi şudur:

> Aynı blind MLP, aynı PPO, reward, domain randomization, task desteği ve eğitim bütçesi altında yalnız task-sampling kuralını değiştirdik; LP-ACRL, handcrafted V4 curriculumuna ve uniform sampling'e kıyasla daha geniş task kapsamına ve/veya aynı kapsama daha az eğitimle ulaştı.

Online estimation bu ana değişken değildir. DreamWaQ ile birleşim, ancak MLP üzerindeki UED sonucu netleştikten sonra ayrı bir kompozisyon deneyi olur.

## 1. Araştırma sorusu, hipotezler ve sınırlar

### Ana soru

> Aynı Go2, Genesis, PPO, reward, domain randomization (DR), observation, policy mimarisi, eğitim bütçesi ve tanımlı task desteği altında; LP-ACRL ile episode-task dağılımını seçmek, V4'ün handcrafted curriculumuna göre öğrenme verimliliğini ve final task coverage'ını artırıyor mu?

### Hipotez

LP-ACRL, hız ve terrain'in çarpım uzayında önceden tanımlanmış tek bir zorluk sırası istemeden, policy'nin o anda öğrenme ilerlemesi gösterdiği hücrelere ağırlık verir. Bu nedenle hedeflenen "yüksek hız + zor terrain" kesişimlerine handcrafted tek-eksenli curriculumdan daha etkili ulaşabilir.

### Bu çalışmanın iddia etmedikleri

- DreamWaQ/HIM/RMA/SysID yöntemlerinin birbirine üstünlüğü.
- Yeni bir reward, DR veya policy mimarisi.
- Makalenin bire bir reproduction'ı.
- PLR, ALP-GMM veya başka bir öğretmen algoritmasının implementasyonunu kullanmak.

## 2. Sabit tutulacak V4 substrate

Ana V5 deneyinde aşağıdakiler bütün kollarda sabittir:

- Simulator ve robot: Genesis + Go2.
- Policy: V4 MLP actor, `45D` noisy proprioceptive observation.
- Critic: mevcut `48D` privileged critic kontratı.
- PPO hiperparametreleri ve toplam eğitim bütçesi: `3000` update.
- Environment sayısı: V4 standardı `4096`.
- Episode süresi: `20 s` / yaklaşık `1000` control step.
- V3 fizik kontratı: friction, added mass, CoM randomization ve episode içi tek mass/CoM switch.
- Mevcut push DR ve diğer nuisance randomization.
- V4 rough-terrain reward seti.
- Terrain üretimi: heightfield substrate ve aynı `terrain_utils` geometrileri korunur; grid, §4'teki makale taksonomisine göre deterministik `6 tip × 4 seviye` builder ile kurulur (default proportions-tabanlı `curiculum()` değil).
- Aynı random seed düzeni, checkpoint aralığı ve dış değerlendirme bankası.

MLP actor terrain height map almaz. Bu bilinçli tercihtir: ilk sonuç, exteroception veya online estimation katkısı değil, episode distribution tasarımının etkisini ölçer.

V4'teki mevcut iki curriculum mekanizması referans baseline'dır:

- Terrain: reset sırasında mesafe-temelli Rudin/game-inspired level promotion.
- Command: bütün kollar aynı ileri hız desteğini kullanır. `handcrafted_v4` kolu iteration `0`da `lin_vel_x=[0,1]`, iteration `500`den sonra `[0,2]` global runner schedule'ına geçer; `uniform`, `lp_acrl` ve `alp` kolları episode-task sampler üzerinden aynı `[0,2] m/s` desteği paylaşır.

V5 UED kollarında bu iki seçim mekanizmasının yerini episode-task sampler alır; yukarıdaki fizik/RL/task-support kontratı değişmez.

## 3. Deney kolları

| Kol | Episode dağıtım kuralı | Rol |
|---|---|---|
| `handcrafted_v4` | Mevcut V4 terrain promotion + command schedule | Gerçek dünya baseline'ı |
| `uniform` | Tanımlı moving task'lar üzerinde eşit olasılık | Curriculum yok kontrolü |
| `lp_acrl` | Signed learning progress üstünde softmax | Ana yöntem |
| `alp` | Absolute learning progress üstünde softmax | Signed LP ablation'ı |

Ana karşılaştırma yalnız MLP ile, eşleştirilmiş en az üç training seed üzerinde yapılır. Bir parallel env istatistiksel seed değildir.

DreamWaQ yalnızca sonraki fazda şu 2×2 için kullanılır:

| Policy | `handcrafted_v4` | `lp_acrl` |
|---|---:|---:|
| MLP | ✓ | ✓ |
| DreamWaQ | ✓ | ✓ |

Bu tablo "online estimation × episode design" etkileşimini verir; V5'in ana headline sonucu değildir.

## 4. Task space: makale taksonomisi ile ilk ciddi deney

İlk ana deney, okunabilirlik için makalenin temiz terrain taksonomisini kullanır (repo'nun 10 ham fiziksel sütunu yerine). Ayrık moving-task uzayı:

\[
\mathcal{T}_{\mathrm{moving}} =
v_x\text{ bin} \times
\text{terrain tipi} \times
\text{terrain zorluğu}.
\]

### Terrain tipleri (6, kategorik)

Makaledeki taksonomi, mevcut `terrain_utils` üreteçleriyle deterministik kurulur:

| Tip | Üreteç | Yön / işaret |
|---|---|---|
| Çıkan merdiven | `pyramid_stairs_terrain` | `step_height > 0` |
| İnen merdiven | `pyramid_stairs_terrain` | `step_height < 0` |
| Yokuş yukarı | `pyramid_sloped_terrain` | `slope > 0` |
| Yokuş aşağı | `pyramid_sloped_terrain` | `slope < 0` |
| Rastgele engebe | `random_uniform_terrain` | — |
| Düz | platform / geometri yok | — |

### Terrain zorluğu (4 seviye, L0→L3)

Her tip için seviye→geometri, makalenin Appendix C tablosuna sadık:

| Tip | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| Merdiven (`step_height`) | 0.05 | 0.10 | 0.15 | 0.20 m |
| Eğim (`\|gradient\|`) | 0.00 | 0.13 | 0.27 | 0.40 |
| Rastgele engebe (`±amplitude`) | 0.02 | 0.045 | 0.07 | 0.10 m |
| Düz | — (dejenere; tek seviye) | | | |

Merdiven `step_width = 0.4 m` sabit. Düz zeminin zorluk ekseni yoktur: eğitimde tek hücre sayılır, fakat §11'deki dünya-sergilemesinde grid tekdüzeliği için 4 özdeş karo olarak çizilir.

### `v_x` binleri (4, genişlik 0.5)

- `[0,0.5)`, `[0.5,1.0)`, `[1.0,1.5)`, `[1.5,2.0]` m/s.
- Tavan `2.0 m/s` bütün kollarda aynıdır. Yüksek hız ile zor terrain kesişimlerinin fiziksel olarak ulaşılamaz olması sonuçtan saklanmaz; SPNTE, success rate ve worst-task raporu bu hücreleri açıkça gösterecektir.
- İlk faz yalnız ileri komutu kapsar; ilgili bin içinde `v_x` uniform örneklenir.

### Sayım

- Eğitim task uzayı: düz tek seviyeye çökünce `5 tip × 4 seviye + 1 düz = 21` terrain-konfig; `× 4 v_x = 84` moving task.
- Grid tekdüze `6 × 4 = 24` terrain karosu istenirse LP, fazlalık düz seviyelerini doğal olarak birleştirir; eğitimde `84`, dünya-sergilemesinde `24` kullanılır.

`v_y` ve yaw ilk fazda bütün kollarda ortak nuisance dağılımından gelir; ilk UED değişkeni yalnız `v_x × terrain`. Direct yaw-rate binleri ve makale ölçeğine yükseltme Faz C'dir (§11).

### Terrain builder notu (önemli)

Default `curiculum()` proportions-tabanlıdır, ayrı bir "düz" tipi üretmez ve merdiven/eğimi işaret üzerinden tek `choice` skalarına gömer → temiz `6 tip × 4 seviye` grid'ini veremez. Bu yüzden `GenesisUEDAdapter` için deterministik **özel builder** yazılır: `(type_index, level_index)` → yukarıdaki tablodaki üreteç + seviye geometrisi, sabit karo yerleşimi. `TaskSpace.fingerprint` bu builder'ın tüm parametrelerini kapsar.

### Standstill kontratı

Mevcut komut kodundaki `zero_cmd_prob=0.4`, task identity ile çelişir: moving task atanmış bir episode daha sonra sıfır komuta dönüşürse return yanlış hücreye yazılır. İlk V5 tasarımı standstill'i LP'nin yönettiği task uzayından ayırır:

\[
p_{\mathrm{train}} = \rho p_{\mathrm{stand}} + (1-\rho)c_j(\zeta),
\qquad \zeta\in\mathcal{T}_{\mathrm{moving}}.
\]

- `\rho` tüm kollarda aynı sabit standing exposure'dır; başlangıçta mevcut marjinal değer olan `0.4` korunur.
- Standing draw per-env yapılır; mevcut batch-wide karar korunmaz.
- Standstill episode'ları LP/ALP task return tahminine girmez, fakat reward/PPO verisine girer.
- Moving episode'larda command, atanmış `v_x` bininin dışına çıkamaz.

Bu, standstill becerisini ihmal etmeden LP sinyalinin anlamını korur.

## 5. LP-ACRL algoritması ve açık implementasyon kontratı

Makalenin çekirdeği:

\[
R_j(\zeta)=\mathbb{E}[R_\tau\mid\zeta,c_j],
\qquad
LP_j(\zeta)=R_j(\zeta)-R_{j-1}(\zeta),
\]

\[
c_{j+1}(\zeta)=
\operatorname{softmax}\!\left(\frac{LP_j(\zeta)}{\beta}\right).
\]

`alp` kolunda yalnız skor değişir:

\[
ALP_j(\zeta)=|R_j(\zeta)-R_{j-1}(\zeta)|.
\]

Makale `\beta`, stage uzunluğu, missing-task davranışı ve return window ayrıntılarını belirtmez. Bu yüzden aşağıdaki V5 kontratı implementasyondan önce dondurulur:

- Başlangıç dağılımı moving task'lar üzerinde uniformdur.
- Stage sınırı PPO iteration veya completed-episode sayısı değil, sabit `global_control_steps`tir.
- İlk stage yalnız `R_0`yı kurar; LP ancak ikinci geçerli ölçümden sonra oluşur.
- Teacher yalnız tamamlanmış ve `valid_for_curriculum=True` episode outcome'larını toplar.
- Training başındaki random episode-length decorrelation nedeniyle kesilmiş ilk episode'lar geçersiz sayılır.
- Her task için `return_sum`, `episode_count`, `R_previous`, `R_current`, `LP` ve observed mask tutulur.
- Bir stage'de yeterli gözlem almayan hücre için sahte reward yazılmaz. Missing-task fallback kuralı testle birlikte açıkça dondurulur.
- Softmax `float64` ve max-shift/logsumexp ile hesaplanır; NaN/Inf fail-fast hatadır.
- EMA, exploration floor, staleness bonusu veya replay mixture ana `lp_acrl` koluna eklenmez. Gerekirse bunlar açık etiketli ayrı ablation'lardır.
- `\beta` yalnız development pilotunda sampler-health ölçütleriyle seçilir ve headline seed'lerden önce sabitlenir.

Sampler-health ölçütleri: finite probabilities, entropy, effective sample size, maximum cell probability, task assignment coverage ve valid completed-outcome coverage.

## 6. Referans kodlardan alınan mimari kararlar

### ALP-GMM'den alınan, alınmayan

ALP-GMM'in yararlı tarafı `sample_task()` ile episode sonundaki `update(task, reward)` ayrımıdır. Teacher sampling logic'i PPO'dan ve environment mapping'den bağımsız durur.

Bu deseni alırız; fakat aşağıdakileri almayız:

- Continuous task space için kNN tabanlı progress hesabı.
- GMM fitting, component seçimi ve continuous task sampling.
- Tek-env `last task` mutable controller modeli.
- Episode sayısına göre fit/update cadence'i.
- Ayrı pickle teacher dump'ı.

V4 finite grid için direct per-cell LP state hem daha doğru hem daha test edilebilirdir.

### PLR'den alınan, alınmayan

PLR'den alınacak desenler:

- Stable integer task identity.
- Env başına açık `active_task_id`.
- Task-indexed persistent score/state dizileri.
- Sampler state'inin checkpoint edilebilir olması.

Alınmayacaklar:

- PPO rollout storage, policy logits, value error ve regret'e bağlı score update.
- Unseen/replay mixture, staging set, bounded level buffer ve eviction.
- Staleness bonusu.
- Rank/power/match transformları.
- `LevelStore` benzeri procedural-level serialization.

V5'in task space'i baştan bilinen finite grid olduğu için bunlar ek fayda sağlamaz ve LP-ACRL deney değişkenini değiştirir.

### Lisans sınırı

- TeachDeepRL/TeachMyAgent ALP-GMM kaynakları MIT lisanslıdır.
- `facebookresearch/dcd` CC BY-NC 4.0 lisanslıdır; özellikle şirket/ticari bağlamında kod taşınmamalıdır.

Her iki referanstan da literal kod taşınmaz. Uygulama clean-room yazılır; alınanlar genel arayüz ve state-management desenleridir.

## 7. Üç katmanlı mimari

```text
TaskSpace / TaskCodec
    stable task_id <-> immutable V4 task specification

EpisodeCurriculum
    task_id ve completed episode outcome bilir;
    sampling, LP/ALP state, stage advance, RNG ve checkpoint state'i yönetir

GenesisUEDAdapter
    task specification'ı gerçek Genesis terrain/type/level/origin ve command
    tensorlerine uygular; old per-env assignment'tan outcome üretir

PPO / policy / critic
    curriculum algoritmasını bilmez
```

### `TaskSpace`

`TaskSpace` immutable source of truth'tür:

```python
class TaskSpace:
    def encode(self, spec: TaskSpec) -> int: ...
    def decode(self, task_id: int) -> TaskSpec: ...
    def decode_batch(self, task_ids: np.ndarray) -> TaskSpecBatch: ...
    def fingerprint(self) -> str: ...
```

Teacher task payload'ını değil yalnız `task_id`yi görür. Terrain type/level/origin ve command-bin mapping yalnız codec/adapter sorumluluğundadır.

### `EpisodeCurriculum`

Hot path 4096 env için Python object listesi değil batch/array tabanlıdır:

```python
@dataclass(frozen=True)
class TaskAssignmentBatch:
    task_ids: np.ndarray          # int64, [N]
    sampler_revision: int
    curriculum_stage: int
    probabilities: np.ndarray     # draw anındaki p(task_id), [N]
    sources: np.ndarray           # bootstrap / lp / alp / standstill

@dataclass(frozen=True)
class EpisodeOutcomeBatch:
    task_ids: np.ndarray
    assigned_revision: np.ndarray
    completion_revision: int
    episodic_returns: np.ndarray
    episode_lengths: np.ndarray
    terminal_reasons: np.ndarray
    valid_for_curriculum: np.ndarray

class EpisodeCurriculum(Protocol):
    def sample(self, count: int, *, global_control_steps: int) -> TaskAssignmentBatch: ...
    def observe(self, outcomes: EpisodeOutcomeBatch) -> None: ...
    def advance(self, global_control_steps: int) -> StageSnapshot | None: ...
    def probabilities(self) -> np.ndarray: ...
    def diagnostics(self) -> Mapping[str, object]: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: Mapping[str, object]) -> None: ...
```

`TaskAssignmentBatch` içindeki revision/probability alanları yalnız convenience değildir: bir episode tamamlandığında hangi sampling dağılımından atandığını kanıtlar.

### `GenesisUEDAdapter`

Adapter'ın görevi:

```python
class GenesisUEDAdapter:
    def collect_outcomes(self, env_ids) -> EpisodeOutcomeBatch: ...
    def assign(self, env_ids, assignments: TaskAssignmentBatch) -> None: ...
    def resample_commands_within_active_bin(self, env_ids) -> None: ...
```

Adapter, per-env `active_task_id`, `active_sampler_revision`, `episode_return` ve `episode_valid` tensorlerini tutar. Teacher Genesis, Torch veya PPO import etmez; adapter LP matematiğini bilmez.

## 8. Reset ve task-provenance state machine'i

Her done/reset için sıralama değişmez:

```text
1. old_task_id / old_assignment_revision'i active per-env tensörden kopyala
2. clipped gerçek episodic return, length ve terminal reason'ı al
3. EpisodeOutcomeBatch'i teacher.observe(...) ile kaydet
4. teacher.sample(...) ile yeni assignment'ları üret
5. TaskSpace.decode_batch(...) ile task payload'ını çöz
6. Genesis adapter terrain_type, terrain_level ve env_origin tensorlerini yaz
7. active_task_id / active_sampler_revision'i yeni assignment'a güncelle
8. Command'i atanan bin içinde örnekle; root state'i yeni origin'de resetle
9. Yeni episode accumulator'larını sıfırla
```

Return her zaman **eski assignment**a yazılır; yeni reset task'ına veya global "last task" değişkenine asla yazılmaz.

Episode içi command resampling yapılacaksa yalnız aktif `v_x` bininin içinden yapılır. Task identity episode boyunca değişmez.

Stage ilerlerken uzun episode'lar sessizce atılmaz. Outcome completion anındaki measurement window'a girer; `assigned_revision` ve `completion_revision` ayrı loglanır. Böylece stage-boundary bleed ölçülür; yüksekse stage süresi mühendislik kararıyla uzatılır, veri saklanarak çözülmez.

## 9. Checkpoint, RNG ve curriculum artifact'leri

Teacher object'i pickle edilmez. PPO checkpoint'ine açık, versiyonlanmış state eklenir:

```python
checkpoint["episode_curriculum"] = {
    "schema_version": 1,
    "algorithm": "uniform" | "lp_acrl" | "alp",
    "task_space_fingerprint": ...,
    "config_fingerprint": ...,
    "stage_index": ...,
    "sampler_revision": ...,
    "stage_start_global_steps": ...,
    "probabilities": ...,
    "previous_returns": ...,
    "current_returns": ...,
    "learning_progress": ...,
    "observed_masks": ...,
    "stage_return_sums": ...,
    "stage_episode_counts": ...,
    "task_assignment_counts": ...,
    "task_completion_counts": ...,
    "transition_occupancy": ...,
    "rng_bit_generator_state": ...,
}
```

Resume'da fingerprint veya schema uyuşmazlığı fail-fast hatadır. Simulator state checkpoint edilmediği için yarım episode'lar restore edilmez; curriculum history ve sampling distribution restore edilir, sonra env'lere yeni task atanır.

Teacher kendi `numpy.random.Generator(PCG64)` RNG'sini kullanır. RNG state checkpoint edilir; `seed=0` geçerlidir. Teacher RNG'si physics/observation RNG'sinden ayrıdır.

Her stage için append-only snapshot üretilir:

- global step, stage ve sampler revision;
- task başına `p`, `R_previous`, `R_current`, LP/ALP, return count;
- assignment count ve completed-episode count;
- assignment-to-completion revision cross-tab;
- entropy, ESS, max probability;
- invalid/discarded outcome sayısı;
- task-headline occupancy ve PPO transition occupancy.

Bu snapshotlar sampling heatmap ve sonradan yapılacak failure analysis için birincil artifact'tir.

## 10. Değerlendirme protokolü

UED değerlendirmesi mevcut online-estimation V4 scorecard'ından ayrıdır. Mevcut terrain eval'in controlled terrain-grid, fixed level ve geometry-hash altyapısı yeniden kullanılır; aggregation ve headline metrikleri UED için ayrı yazılır.

Her yöntem için:

- Aynı fixed `84`-task validation grid.
- Aynı `[0,2] m/s` command bank.
- Aynı terrain geometry seed'leri ve hash kontrolü.
- Aynı DR pinleri.
- Aynı evaluation seed'leri.
- Aynı checkpoint değerlendirme aralığı; başlangıçta her `200` PPO update.

Her checkpoint curriculumun canlı training dağılımında değil, yukarıdaki ortak fixed validation bank üzerinde değerlendirilir. Ana checkpoint, aşağıda tanımlanan SPNTE skorunu en aza indiren `best_spnte.pt`dir. `best_tracking.pt` ana seçim değildir; `model_3000.pt` yalnız eğitim sonu/provenance artifact'i olarak saklanır.

### Primary metrikler

- Mean SPNTE.
- Final task success rate.
- Sample efficiency: önceden seçilmiş iteration'larda success-rate curve/AUC.
- Worst-%10 task CVaR.

### Secondary metrikler

- Episodic return, fall probability ve survival length.
- Terrain/command slice heatmap'leri.
- LP/probability heatmap'lerinin zaman serisi.
- Entropy/ESS ve task coverage.
- Assignment probability ile gerçek PPO transition occupancy farkı.

Zor task'lar erken düştüğü için episode başına örnekleme olasılığı ile PPO buffer'daki transition oranı aynı değildir; ikisi ayrı raporlanır.

### SPNTE: Stability-Penalized Normalized Tracking Error

Makaledeki EPTE-SP'nin yüzde hata paydası sıfır komutta açık değildir. V5, aynı stability fikrini fakat açık normalizasyonla kullanır:

\[
e_t^{lin} =
\operatorname{clip}
\left(
\frac{|v_{x,t}^{cmd}-v_{x,t}^{base}|}{v_{\mathrm{scale}}},
0,
1
\right),
\]

\[
\mathrm{SPNTE}_{lin} =
\frac{
\sum_{t<k_f} e_t^{lin} + (K-k_f)
}{K}.
\]

Burada:

- `K=1000`, sabit evaluation episode uzunluğudur.
- `k_f`, ilk düşüş adımıdır; düşme yoksa `k_f=K`.
- `v_scale`, anlık komut büyüklüğü değil, **o kampanyanın dondurulmuş linear command support ölçeğidir** ve dinamik okunur: `v_scale = max(|lin_vel_x_min|, |lin_vel_x_max|)`. V5 forward-only `[0,2]` için `v_scale=2.0`; V4 `[-1,1]` için `v_scale=1.0`; V4 sprint-schedule `[-1,2]` için `v_scale=2.0`. Değer eval config'inin komut bankasından türetilir, hardcode edilmez, ve artifact'e `spnte_v_scale` olarak yazılır. `v_x^{cmd}=0` iken bölme problemi oluşmaz çünkü payda komut değeri değil support ölçeğidir.
- İlk düşüşten sonraki her adım `1.0`, yani maksimum hata sayılır. Auto-reset sonrasındaki yeni episode performansı aynı örneğin skorunu iyileştiremez.
- Düşme yoksa SPNTE yalnız normalize tracking hatasına eşittir.
- Düşük SPNTE daha iyidir.
- **Default metrik SPNTE'dir; eski metrikler (`tracking_lin_err`, `fall_rate`, `mean_return`, ...) crosscheck için korunur ve aynı koşumda birlikte hesaplanır** (§14.2). Bu yüzden eval harness'ının `auto_reset=True` davranışı bozulmaz: eski metrikler tüm horizon'u tüketmeye devam eder, SPNTE ise aynı stream üzerinde first-fall semantiğiyle ayrıca birikir.

Yaw task ekseni ilk fazda curriculum değişkeni olmadığı için checkpoint seçimine karıştırılmaz. Aynı stability-penalty mantığıyla `SPNTE_yaw` ayrıca loglanır; Faz C'de yaw task ekseni eklendiğinde açık bir primary metriğe dönüşür.

Bu metrik EPTE-SP'den esinlenir fakat farklı normalizasyon kullandığı için makaledeki adla sunulmaz.

### Fixed validation bank ve agregasyon

- Validation bank eğitim task desteğindeki `84` hücrenin tamamını kapsar: `(5 tip × 4 seviye + 1 düz) × 4 v_x bin`.
- Her hücre `48` deterministic replica ile ölçülür; toplam validation environment sayısı `84 × 48 = 4032` olur.
- Her replica için bin içindeki command draw önceden üretilir ve bütün checkpoint/yöntemlerde aynen tekrar kullanılır.
- Checkpoint seçme bankası başlangıçta `validation_seed=31001`, final held-out banka `eval_seed=41001` kullanır. Seed'ler ve geometry hash'leri headline koşularından önce config'te dondurulur.
- Önce her replica için SPNTE hesaplanır; sonra hücre ortalaması alınır.
- Checkpoint'in ana skoru, `84` hücre ortalamasının eşit ağırlıklı ortalamasıdır. Çok sayıda kolay örnek zor hücreleri ezemez.
- Worst-%10 task CVaR, fall rate, survival fraction ve success rate ayrıca raporlanır.

### Checkpoint seçimi

Her `200` PPO update'de ortak fixed validation bank çalıştırılır:

```text
model_200.pt, model_400.pt, ..., model_3000.pt
```

Ana seçim kuralı:

```text
en düşük 84-task macro-mean SPNTE → best_spnte.pt
```

Eşitlik/tie-break sırası:

1. Daha düşük worst-%10 task SPNTE.
2. Daha düşük fall rate.
3. Daha yüksek success rate.
4. Daha erken iteration.

Ana skorlar `1e-6` mutlak tolerans içinde eşitse tie-break uygulanır; aksi durumda doğrudan daha düşük macro-mean SPNTE kazanır.

Checkpoint içine `selection_metric=spnte_v1`, skor bileşenleri, validation-bank fingerprint'i, geometry hash'leri, support ölçeği ve selected iteration yazılır. Resume sonrası önceki `best_spnte` anahtarı ve artifact'i korunur.

Başlangıç success kuralı:

- en az `900/1000` step survival;
- `SPNTE_lin <0.30`.

`SPNTE_yaw <0.30` Faz C'de yaw task ekseni açıldığında success kontratına eklenir. Hücre başarısı tek rollout'a göre değil, önceden belirlenmiş replica-success oranına göre tanımlanır. Eşik sonuç görüldükten sonra değiştirilmez.

Held-out ana değerlendirme aynı task hücreleri üzerinde yeni deterministic terrain geometry seed'leri ve command draw'ları kullanır. Support dışı hız/severity yalnız ikincil OOD deneyidir.

## 11. Uygulama ve deney fazları

### Faz 0 — Spesifikasyonu kilitle

Koddan önce şu sayılar/kurallar tek bir config ve test dokümanında dondurulur:

- Task binleri ve task-space fingerprint.
- Standstill mixture `rho`.
- Stage control-step uzunluğu.
- Missing-task/minimum-count kuralı.
- `beta` development prosedürü ve seçilen değer.
- SPNTE `v1` formülü, `2.0 m/s` support ölçeği ve ilk-düşüş indeks semantiği.
- Fixed `84 × 48` validation matrix, validation/final seed'leri ve success threshold.
- `best_spnte.pt` seçim ve tie-break kuralı.
- Curriculum checkpoint schema.

### Faz A — Saf teacher ve 8-task Genesis smoke

- `2 v_x bin × 2 terrain tipi × 2 seviye = 8` task.
- `64–128` env.
- Sentetik outcome ile LP/ALP probability shift unit testleri.
- Gerçek Genesis reset ile type/level/origin/command-bin doğrulaması.
- Eğitim sonucu iddiası yok.

### Faz B — makale-taksonomisi MLP UED benchmark

- Task uzayı §4: `(5 tip × 4 seviye + 1 düz) × 4 v_x = 84` moving task (yaw nuisance).
- `handcrafted_v4`, `uniform`, `lp_acrl`, `alp`.
- Aynı MLP/PPO/DR/reward/task support/budget.
- En az üç paired training seed.
- `3000` PPO update.
- Ortak fixed validation bank üzerinde SPNTE ile seçilen `best_spnte.pt` primary endpoint.
- `model_3000.pt` yalnız eğitim sonu duyarlılık/provenance sonucu olarak ayrıca raporlanır.
- Bu faz, V5'in ana bilimsel sonucu olur.

### Faz C — Yaw ekseni + makale ölçeğine yükseltme

Faz B başarıyla tamamlandıktan sonra, aynı 6-tip taksonomisi üstünde:

- Direct yaw rate için `heading_command=False`; `|ω_z|` binleri eklenir (makale: `[0,3.0]`, 6 seviye).
- Ölçek makaleye yaklaştırılır, ör. `6 tip × 4 seviye × 4 v_x × 6 |ω_z|`; istenirse `v_x` tavanı ve seviye sayısı da makaleye doğru genişletilir.
- Genesis/Go2 terrain parametrizasyonu açıkça belgelenir.
- Sonuç paper-inspired olarak sunulur; bire bir reproduction iddiası yapılmaz.

### Faz D — DreamWaQ birleşimi

Yalnız Faz B'de LP-ACRL'in kazanımı netse uygulanır. MLP/DreamWaQ × handcrafted/LP-ACRL 2×2 tablosu, UED'nin online estimation ile tamamlayıcılığını ölçer.

Push magnitude, friction frontier ve command-transition sertliği ancak bundan sonra değerlendirilir.

## 12. Kabul kriterleri

### Saf algoritma testleri

- `TaskSpace.encode/decode` bire birdir.
- Uniform sampling istatistiksel olarak uniformdur.
- Pozitif LP ilgili hücrenin olasılığını artırır.
- Negatif LP, `alp`de artabilir fakat signed `lp_acrl`de artmaz.
- Softmax finite'dir ve toplamı `1`dir.
- Aynı RNG state aynı draw sequence'ini üretir.
- `state_dict/load_state_dict` sonrası draw sequence kesintisiz devam eder.
- Teacher modülü Genesis, Torch, PPO veya policy import etmez.

### Environment/Genesis testleri

- Her sampled command aktif binin içindedir; episode içi resampling bin dışına çıkmaz.
- Terrain type, level ve origin tek task assignment'tan gelir.
- Return eski `active_task_id` ve eski revision'a yazılır.
- Batch reset'te env'lerin task identity'leri karışmaz.
- Başlangıçtaki kesik episode'lar curriculum outcome'u sayılmaz.
- Eval rollout'u teacher state hash'ini değiştirmez.
- Stage boundary cross-revision sayıları snapshot'ta doğru görünür.

### Kontrat ve deney testleri

- Dört kolun PPO, reward, DR, actor/critic, task support ve budget kontratı eşittir; allowlist dışı fark fail eder.
- Curriculum checkpoint başka task/config/algorithm fingerprint'iyle yüklenmez.
- Geometry hash, command bank ve eval seed bütün yöntemlerde aynıdır.
- Düşme olmayan sabit-hatalı sentetik episode'da SPNTE normalize tracking hatasına eşittir.
- İlk düşüşten sonraki auto-reset adımları SPNTE'yi iyileştirmez; kalan horizon maksimum hata sayılır.
- `v_x^{cmd}=0` dahil bütün command desteğinde SPNTE finite ve `[0,1]` aralığındadır.
- Her checkpoint tam `84 × 48` validation bank üzerinde ölçülmeden seçim yapılamaz.
- En düşük macro-mean SPNTE checkpoint'i `best_spnte.pt` olur; tie-break yalnız `1e-6` tolerans içinde uygulanır.
- Resume, `best_spnte` seçim anahtarını ve artifact'ini kaybetmez.
- Assignment distribution ve PPO transition occupancy ayrı kaydedilir.
- Genesis smoke gerçek type/level/origin geçişini doğrular.

## 13. Son konumlandırma

Bu damar şu adla sunulur:

> **V5 / UED: Automatic episode-distribution design on the frozen V4 locomotion substrate**

İlk ana iddia yalnız curriculum hakkındadır. DreamWaQ ile daha iyi sonuç alınırsa, bu ikinci ve ayrı bir "iyi episode design + online estimation" birleşim sonucudur.

## 14. Netleştirmeler ve dondurulmuş kararlar (2026-07-23)

Bu bölüm, implementasyon öncesi açık kalan noktaları dondurur. Çelişki olursa bu
bölüm önceki bölümleri override eder. Uygulama, `lpacr/subplans/` altındaki
konu-başına subagent spec'lerine göre yürütülür; her subplan buradaki kararları
referans alır.

### 14.0 Ön koşul — versiyon kontrolü

Repo `genesis-wp/LeggedGym-Ex` bir git deposudur; `lpacr/` artık bu deponun
içine taşınmıştır (dashboard, plan, article, notlar dahil). Herhangi bir
subagent turu başlamadan önce temiz bir baseline commit'i alınır. Her subplan
kendi çalışmasını ayrı branch'te yapar; çekirdek paylaşılan dosyalara
(`legged_robot.py`, `metrics.py`) dokunan subplanlar (özellikle §Topic-03)
mutlaka flag arkasına alır ki v3/v4 bit-for-bit korunsun.

### 14.1 SPNTE V4'te de çalışır; ölçek dinamiktir

- SPNTE hem V5 hem V4 (ve genel olarak "bundan sonra") eval'lerinde hesaplanır.
- Normalizasyon paydası `v_scale` **hardcode edilmez**; aktif eval config'inin
  linear command bankasından `v_scale = max(|lin_vel_x_min|, |lin_vel_x_max|)`
  olarak türetilir ("kaç ise o olsun"). V4 `[-1,1]` → `1.0`, V5 `[0,2]` → `2.0`.
- Seçilen `v_scale` artifact'e `spnte_v_scale` alanı olarak yazılır ki farklı
  kampanyaların SPNTE sayıları yanlışlıkla kıyaslanmasın.

### 14.2 Default metrik SPNTE; eski metrikler crosscheck olarak korunur

- Yöntem sıralaması / checkpoint seçimi V5'te SPNTE'ye göredir (`best_spnte.pt`).
- Eski scorecard metrikleri (`tracking_lin_err`, `tracking_lin_rmse`,
  `fall_rate`, `mean_return`, `mean_ep_len`, ...) **silinmez**; aynı koşumda
  SPNTE ile birlikte hesaplanıp raporlanır. Böylece eski↔yeni tutarlılığı
  (crosscheck) her checkpoint'te görülebilir.
- V4_terrain kampanyası append-only kalır: SPNTE oraya **ek sütun** olarak
  eklenir, mevcut sayılar yeniden koşulmaz ve değiştirilmez.

### 14.3 SPNTE eval semantiği: accumulator, auto_reset açık

- "Eski ve yeni eval aynı anda çalışsın" gereği, harness'ın `auto_reset=True`
  davranışı korunur (eski metrikler tüm horizon'u tüketir).
- Bu nedenle SPNTE, **auto_reset'i kapatmaz**; onun yerine `MetricAccumulator`'a
  per-env first-fall state'i eklenir:
  - her env için `first_fall_step` (ilk `done & ~time_out`), başta `-1`;
  - normalize hata yalnız `t < k_f` için birikir; `k_f` sonrası hiçbir yeni
    episode katkısı sayılmaz;
  - `compute()`'te kalan horizon `(K - k_f)` maksimum hata (`1.0`) ile doldurulur.
- Böylece §12'deki "auto-reset adımları SPNTE'yi iyileştirmez" kriteri, harness'ı
  bozmadan sağlanır. (Alternatif "tek-episode, auto_reset kapalı" tasarımı
  reddedildi çünkü eski metriklerin eşzamanlı koşumunu bozardı.)

### 14.4 Standstill: per-env, çekirdek env flag arkasında

- Mevcut `_resample_commands` (`legged_robot.py`) zero-command kararını
  **tek skaler** `np.random.rand()` ile tüm batch'e uyguluyor. UED kontratı bunu
  per-env yapılmasını gerektirir (§4 standstill).
- Bu değişiklik `commands.per_env_standstill` (veya eşdeğer) flag'i arkasına
  alınır; flag kapalıyken v3/v4 davranışı birebir korunur, yalnız UED kolları
  açar.
- Standstill episode'ları LP/ALP task-return tahminine **girmez** ama PPO/reward
  verisine girer (`valid_for_curriculum=False`).

### 14.5 Command-curriculum ve command_schedule UED kollarında kapalı

- `_update_command_curriculum` (tracking-reward'a bağlı otomatik aralık
  genişletme) ve config'teki `command_schedule`, dağıtımı runtime'da mutasyona
  uğratır. UED kollarında (`uniform`/`lp_acrl`/`alp`) sampler dağıtımın tek
  sahibidir; bu iki mekanizma bu kollarda **kapatılır**.
- `handcrafted_v4` kolunda ikisi de mevcut V4 davranışıyla açık kalır (o kol
  gerçek-dünya baseline'ı). Bu asimetri kasıtlıdır ve kontrat testinde
  allowlist'e yazılır.

### 14.6 Terrain teleport modeli (adapter doğrulaması)

- 84-task training grid'i, mevcut `taxonomy_showcase` builder'ının kanıtladığı
  desenle **statik** kurulur; env'ler reset'te doğru tile origin'ine ışınlanır.
  `GenesisUEDAdapter.assign` heightfield'i yeniden kurmaz, yalnız per-env
  `env_origin`/`terrain_level`/command-bin yazar.
- Faz A smoke'ın ilk doğrulaması: reset anında bir env'i keyfi tile'a taşımanın
  heightfield rebuild olmadan, doğru type/level/origin ile çalıştığı.

### 14.7 play.py `--terrain taxonomy` durumu

- Taksonomi exhibit (6 tip × 4 seviye) play.py + terrain.py'de zaten mevcut
  (`taxonomy_showcase`, `build_taxonomy_label_map`, `test_taxonomy_showcase.py`).
- Bu, §14.6'daki eğitim/eval builder'ından **ayrıdır** (görsel sergi vs. task
  grid). Eğitim builder'ı taksonomi geometrisini paylaşır ama kendi
  `TaskSpace.fingerprint`'i ve deterministik tile yerleşimi ile.

### 14.8 Konu ayrımı (subagent decomposition)

Plan aşağıdaki bağımsız iş kalemlerine bölünür; detay ve interface kontratları
`lpacr/subplans/`'ta. Bağımlılık sırası:

| Topic | Dosya | Bağımlılık |
|---|---|---|
| 01 SPNTE metrik + eval wiring (V4+V5) | `01_spnte_metric.md` | yok — ilk başlar |
| 02 Clean-room UED teacher (TaskSpace/Curriculum) | `02_ued_teacher_core.md` | yok |
| 03 Genesis entegrasyonu + provenance state machine | `03_genesis_integration.md` | 02 |
| 04 Validation bank + `best_spnte.pt` seçimi | `04_validation_and_checkpoint.md` | 01 |
| 05 Kollar/config + kontrat testleri + Faz B | `05_arms_config_and_fazB.md` | 02,03,04 |

01 ve 02 tamamen paralel ve düşük risklidir (çekirdek env'e dokunmaz). 03 asıl
riskli entegrasyondur ve en dar allowlist ile en sona bırakılır.
