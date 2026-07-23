# Topic 05 — Deney kolları/config + kontrat testleri + Faz B orchestration

**Bağımlılık:** Topic 02 (teacher), 03 (Genesis entegrasyonu), 04 (validation).
En son çalışır; hepsini entegre eder.
**Referans:** `../solun_plani.md` §2 (sabit substrate), §3 (kollar), §11 (Faz 0/B),
§12 (kontrat testleri), §14.5.

## Amaç

Dört UED kolunu (`handcrafted_v4`, `uniform`, `lp_acrl`, `alp`) tek bir V5 task
ailesinde kur; PPO/reward/DR/actor-critic/task-support/budget kontratının kollar
arası eşitliğini kontrat testiyle garanti et; Faz 0 config-freeze dokümanını yaz;
Faz B eğitimini (≥3 paired seed, 3000 update) orkestre et.

## Dokunulacak / yeni dosyalar

- `legged_gym/envs/go2/go2_v5_config.py` (yeni) — V4 substrate'i sabit tutan V5
  cfg'leri; kol seçimi `curriculum.algorithm ∈ {handcrafted_v4, uniform, lp_acrl,
  alp}` flag'iyle (dört ayrı env yerine tek aile + flag tercih edilir).
- `legged_gym/envs/__init__.py` — `go2_v5_*` registration'ları (mevcut `go2_v4_*`
  deseni, satır ~233).
- `tests/test_v5_training_contract.py` (yeni) — allowlist kontrat testi.
- `lpacr/faz0_freeze.md` (yeni) — Faz 0 dondurma dokümanı.
- Faz B launcher (mevcut v4 seed-pair launcher desenini izle:
  `scripts/run_v4_seed1_pair.sh`).

## Sabit V5 substrate (§2) — kollar arası DEĞİŞMEZ

Genesis+Go2, V4 MLP actor (45D noisy proprio), 48D privileged critic, PPO HP,
3000 update, 4096 env, 20s/~1000 step episode, V3 fizik kontratı (friction/added
mass/CoM + episode-içi tek switch), mevcut push DR, V4 rough-terrain reward,
heightfield substrate. MLP actor terrain height map **almaz** (bilinçli).

## Kollar (§3)

| Kol | Episode dağıtım kuralı |
|---|---|
| `handcrafted_v4` | Mevcut V4 terrain promotion + command schedule (baseline) |
| `uniform` | Moving task'larda eşit olasılık |
| `lp_acrl` | Signed LP softmax (ana yöntem) |
| `alp` | `|LP|` softmax (ablation) |

- `handcrafted_v4`: command_schedule + `_update_command_curriculum` **açık**
  (mevcut V4). `handcrafted_v4` iteration 0'da `lin_vel_x=[0,1]`, 500'den sonra
  `[0,2]` (§2).
- `uniform`/`lp_acrl`/`alp`: episode-task sampler üzerinden ortak `[0,2] m/s`
  destek; command_schedule + command-curriculum **kapalı** (§14.5).

## Kontrat testi (§12) — allowlist

- Dört kolun PPO, reward, DR, actor/critic, task-support ve budget kontratı eşit;
  **allowlist dışı fark fail eder.** Allowlist'e yazılan kasıtlı farklar:
  - episode-task sampler (yok/uniform/lp_acrl/alp),
  - command_schedule + command-curriculum durumu (yalnız `handcrafted_v4` açık),
  - standstill mekanizması (batch-wide vs per-env).
- Curriculum checkpoint başka task/config/algorithm fingerprint'iyle yüklenmez.
- Geometry hash / command bank / eval seed bütün kollarda aynı.

## Faz 0 freeze dokümanı (`lpacr/faz0_freeze.md`, §11 Faz 0)

Koddan önce dondur: task binleri + task-space fingerprint; standstill `rho`; stage
control-step uzunluğu; missing-task/min-count kuralı; `beta` prosedürü + seçilen
değer; SPNTE v1 formülü + `v_scale` + first-fall semantiği; `84×48` matrix +
validation/final seed + success eşiği; `best_spnte.pt` seçim/tie-break;
curriculum checkpoint schema. (Çoğu Topic 01–04'te zaten dondu; bu doküman tek
yerde toplar.)

## Faz B (§11 Faz B)

- Task uzayı §4: `84` moving task.
- Dört kol, aynı MLP/PPO/DR/reward/task-support/budget.
- ≥3 paired training seed (bir parallel env istatistiksel seed **değildir**).
- 3000 PPO update; her 200'de fixed bank (Topic 04) → `best_spnte.pt` primary
  endpoint. `model_3000.pt` yalnız provenance.
- `beta` development pilotu (sampler-health) headline seed'lerden **önce** koşulur
  ve sabitlenir.

## Done tanımı

- `go2_v5_*` register; `test_v5_training_contract.py` yeşil (allowlist enforced).
- `lpacr/faz0_freeze.md` tüm dondurulmuş sayıları tek yerde içerir.
- Faz B launcher hazır; en az 1 seed ile uçtan uca smoke (kısa update) geçer.
- Bu, V5'in ana bilimsel sonucunu üretecek pipeline'dır (§11 Faz B).
