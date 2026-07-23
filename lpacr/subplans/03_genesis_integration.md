# Topic 03 — Genesis entegrasyonu + task-provenance state machine

**Bağımlılık:** Topic 02 (interface'leri tüketir).
**Risk:** YÜKSEK — çekirdek env dosyalarına dokunur. Tüm değişiklikler flag
arkasında; flag kapalıyken v3/v4 birebir korunur.
**Referans:** `../solun_plani.md` §7 (adapter), §8 (reset state machine), §14.4–14.7.

## Amaç

`EpisodeCurriculum`'ı (Topic-02) gerçek Genesis/Go2 env'ine bağla. Reset'te
per-env terrain teleport + command-bin sampling + outcome toplama; task-provenance
state machine; per-env standstill; command-curriculum-off. Faz A 8-task smoke ile
doğrula.

## Dokunulacak dosyalar

- `legged_gym/utils/ued/genesis_adapter.py` (yeni) — `GenesisUEDAdapter`.
- `legged_gym/utils/terrain.py` — deterministik 84-task training builder
  (`taxonomy_showcase` desenini genişlet; ayrı fingerprint'li).
- `legged_gym/envs/base/legged_robot.py` — reset/command hook'ları, **flag
  arkasında** (`per_env_standstill`, `command_curriculum_enabled`).
- `legged_gym/envs/base/legged_robot_config.py` — yeni flag default'ları (kapalı).

## GenesisUEDAdapter (§7)

```python
class GenesisUEDAdapter:
    def collect_outcomes(self, env_ids) -> EpisodeOutcomeBatch: ...
    def assign(self, env_ids, assignments: TaskAssignmentBatch) -> None: ...
    def resample_commands_within_active_bin(self, env_ids) -> None: ...
```

- Per-env tensor tutar: `active_task_id`, `active_sampler_revision`,
  `episode_return`, `episode_valid`.
- `assign`: `TaskSpace.decode_batch` sonucundan `terrain_type`/`terrain_level`/
  `env_origin` ve command-bin yazar. Heightfield'i **yeniden kurmaz** (statik grid,
  teleport — §14.6).
- Adapter LP matematiğini bilmez; teacher Genesis/Torch bilmez.

## Reset & provenance state machine (§8, sıra DEĞİŞMEZ)

```
1. old_task_id / old_assignment_revision'i kopyala
2. clipped gerçek episodic return, length, terminal reason al
3. teacher.observe(EpisodeOutcomeBatch)   # return ESKİ assignment'a yazılır
4. teacher.sample(...) -> yeni assignment
5. TaskSpace.decode_batch(...)
6. adapter: terrain_type/level/env_origin yaz
7. active_task_id/revision güncelle
8. command'i atanan bin İÇİNDE örnekle; root state'i yeni origin'de resetle
9. episode accumulator'larını sıfırla
```

- Return **her zaman eski assignment'a** yazılır; asla yeni reset task'ına veya
  global "last task"a değil.
- Episode-içi command resampling yalnız aktif `v_x` bini içinden; task identity
  episode boyunca sabit.
- Stage boundary'de uzun episode'lar atılmaz; `assigned_revision` vs
  `completion_revision` ayrı loglanır (bleed ölçülür).

## 84-task deterministik builder (§4 "terrain builder notu")

- Default `curiculum()` proportions-tabanlı → temiz `6 tip × 4 seviye` grid
  veremez; ayrı builder yazılır.
- `(type_index, level_index)` → taksonomi üreteci + seviye geometrisi (step/slope/
  amplitude tabloları taksonomi ile aynı: `TAXONOMY_STEP_HEIGHTS` vb.).
- Düz tip tek-seviyeye çöker (eğitimde 21 terrain-config × 4 v_x = 84).
- Sabit tile yerleşimi; `TaskSpace.fingerprint` builder'ın tüm parametrelerini
  kapsar. Bu builder taksonomi **exhibit'inden ayrıdır** (§14.7).

## Standstill (§14.4) — flag arkasında

- Mevcut `_resample_commands` (`legged_robot.py:534`) zero-command'ı tek skaler
  `np.random.rand()` ile tüm batch'e uygular. `per_env_standstill=True` iken
  per-env draw yapılır.
- `p_train = rho * p_stand + (1-rho) * c_j(zeta)`, `rho=0.4` (mevcut marjinal).
- Standstill episode `valid_for_curriculum=False`; LP/ALP'ye girmez, PPO/reward'a
  girer.
- Flag kapalı → mevcut batch-wide davranış birebir.

## Command-curriculum off (§14.5) — flag arkasında

- `_update_command_curriculum` ve config `command_schedule`, UED kollarında
  kapalı (`command_curriculum_enabled=False`). `handcrafted_v4`'te açık.

## Faz A smoke (§11 Faz A)

- `2 v_x × 2 tip × 2 seviye = 8` task, `64–128` env.
- Sentetik outcome ile LP/ALP probability shift (Topic-02 testleriyle örtüşür).
- Gerçek Genesis reset ile type/level/origin/command-bin doğrulaması.
- **İlk doğrulama:** reset'te env'i keyfi tile'a taşımanın heightfield rebuild
  olmadan doğru type/level/origin ürettiği (§14.6). Bu tutmuyorsa builder/teleport
  modelini burada düzelt — sonraki her şey buna bağlı.

## Kabul testleri (§12 environment/Genesis)

- Her sampled command aktif binin içinde; episode-içi resampling bin dışına
  çıkmaz.
- Terrain type/level/origin tek assignment'tan.
- Return eski `active_task_id`+revision'a yazılır.
- Batch reset'te task identity'ler karışmaz.
- Başlangıç kesik episode'ları curriculum outcome'u sayılmaz.
- Eval rollout'u teacher state hash'ini değiştirmez.
- Stage boundary cross-revision sayıları snapshot'ta doğru.
- **flag kapalı → `test_v3_training_contract.py` / `test_v4_training_contract.py`
  birebir yeşil** (regression guard).

## Done tanımı

- Faz A 8-task Genesis smoke geçer; teleport/type/level/origin doğrulandı.
- Flag kapalıyken v3/v4 kontrat testleri değişmeden yeşil.
- `GenesisUEDAdapter`, Topic-02 interface'ine tam oturur.
