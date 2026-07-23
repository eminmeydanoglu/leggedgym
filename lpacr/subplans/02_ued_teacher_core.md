# Topic 02 — Clean-room UED teacher core (TaskSpace / EpisodeCurriculum)

**Bağımlılık:** yok. 01 ile tamamen paralel. Çekirdek env'e / Genesis / Torch /
PPO'ya **dokunmaz, import etmez.**
**Referans:** `../solun_plani.md` §5, §6, §7, §9, §12 (saf-algoritma testleri).

## Amaç

LP-ACRL / ALP / uniform episode-task sampler'ını saf Python + numpy ile,
clean-room olarak yaz. Finite 84-hücre grid için per-cell LP state. Test edilebilir,
checkpoint edilebilir, RNG-deterministik. Bu paket episode dağıtımının beynidir;
env mapping ve terrain'i bilmez (o §Topic-03).

## Lisans sınırı (kritik)

- TeachMyAgent/ALP-GMM (MIT) ve `facebookresearch/dcd` (CC BY-NC 4.0)
  **kaynaklarından literal kod taşınmaz.** Yalnız genel arayüz/state-management
  deseni alınır (sample/update ayrımı, integer task identity, task-indexed diziler).
- kNN progress, GMM fit, LevelStore, replay/staleness/rank transformları **alınmaz**
  (§6).

## Yeni modül

Öneri: `legged_gym/utils/ued/` (yeni paket), Genesis/Torch import etmeyen saf
modüller:

- `task_space.py` — `TaskSpace`, `TaskSpec`, `TaskSpecBatch`.
- `episode_curriculum.py` — `EpisodeCurriculum` protokolü + `Uniform`, `LPACRL`,
  `ALP` implementasyonları; `TaskAssignmentBatch`, `EpisodeOutcomeBatch`,
  `StageSnapshot`.
- `checkpoint.py` — schema v1 serialize/deserialize + fingerprint kontrolü.

## Interface kontratı (Topic-03 buna bağlanır)

`../solun_plani.md §7`'deki imzalar **aynen** kullanılır:

```python
class TaskSpace:
    def encode(self, spec: TaskSpec) -> int: ...
    def decode(self, task_id: int) -> TaskSpec: ...
    def decode_batch(self, task_ids) -> TaskSpecBatch: ...
    def fingerprint(self) -> str: ...

class EpisodeCurriculum(Protocol):
    def sample(self, count, *, global_control_steps) -> TaskAssignmentBatch: ...
    def observe(self, outcomes: EpisodeOutcomeBatch) -> None: ...
    def advance(self, global_control_steps) -> StageSnapshot | None: ...
    def probabilities(self) -> np.ndarray: ...
    def diagnostics(self) -> Mapping[str, object]: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state) -> None: ...
```

`TaskAssignmentBatch` / `EpisodeOutcomeBatch` alanları §7'deki gibi (batch/array
tabanlı, 4096 env için Python-object listesi yok).

## Frozen algoritma kararları (§5)

- Başlangıç dağılımı moving task'lar üzerinde uniform.
- Stage sınırı = sabit `global_control_steps` (PPO iteration/episode sayısı değil).
- İlk stage yalnız `R_0`; LP ikinci geçerli ölçümden sonra.
- Yalnız `valid_for_curriculum=True` outcome'lar toplanır; başlangıç decorrelation
  kesik episode'ları geçersiz.
- Per-task: `return_sum`, `episode_count`, `R_previous`, `R_current`, `LP`,
  observed mask.
- Yeterli gözlem almayan hücreye **sahte reward yazılmaz**; missing-task fallback
  kuralı testle birlikte dondurulur (öneri: gözlemsiz hücre LP'ye girmez, olasılığı
  önceki stage'den taşınır — kararı testte sabitle).
- Softmax `float64` + max-shift/logsumexp; NaN/Inf fail-fast.
- `lp_acrl`: signed LP softmax. `alp`: `|LP|` softmax. EMA/floor/staleness/replay
  ana kollara **eklenmez**.
- `beta` yalnız development pilotunda sampler-health ile seçilir, headline'dan önce
  sabitlenir (bu subplan `beta`'yı parametre bırakır, değerini seçmez).
- Teacher kendi `numpy.random.Generator(PCG64)` RNG'sini kullanır; `seed=0`
  geçerli; RNG state checkpoint'lenir.

## Checkpoint schema (§9)

`checkpoint.py`, §9'daki dict'i üretir/yükler (`schema_version=1`, algorithm,
`task_space_fingerprint`, `config_fingerprint`, stage/revision, probabilities,
previous/current returns, LP, observed_masks, sayaçlar, transition_occupancy,
`rng_bit_generator_state`, ...). Fingerprint/schema uyuşmazlığı **fail-fast**.

## Sampler-health diagnostics (§5)

`diagnostics()`: finite probabilities, entropy, effective sample size (ESS),
max cell probability, task assignment coverage, valid completed-outcome coverage.

## Kabul testleri (§12 saf-algoritma)

- `encode/decode` bire bir; `decode_batch` tutarlı.
- Uniform sampling istatistiksel olarak uniform.
- Pozitif LP → ilgili hücre olasılığı artar.
- Negatif LP → `alp`'de artabilir, signed `lp_acrl`'de artmaz.
- Softmax finite ve toplam 1.
- Aynı RNG state → aynı draw sequence.
- `state_dict`/`load_state_dict` sonrası draw sequence kesintisiz.
- **Modül Genesis/Torch/PPO/policy import etmez** (import-guard testi).
- Missing-task fallback: dondurulan kural testte deterministik.

## Done tanımı

- `legged_gym/utils/ued/` paketi + `tests/test_ued_teacher.py` yeşil.
- Sıfır Torch/Genesis import (test doğrular).
- Interface, Topic-03'ün beklediği imzalarla birebir.
