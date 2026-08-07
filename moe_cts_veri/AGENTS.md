# MoE-CTS veri arşivi — veri raporu

Bu klasör, `go2_moects` (MoE-CTS) kolu üzerinde toplanan **bütün ölçüm verilerini**
tek giriş noktasında toplar. **İçindeki her şey göreli symlink'tir** — dosyaların
gerçek yeri repodaki orijinal konumları; burada kopya yok, bir dosyayı düzenlersen
orijinali düzenlemiş olursun.

Kapsam: `go2_moects` / `ActorCriticMoECTS` / `PPO_MOE_CTS`.
Birincil koşu: `logs/go2_moects/Aug03_12-01-45_moe_cts_genesis` (Go2, 8 uzman, 32-D latent).
Karşılaştırma tarafı: wty-yy `go2_rl_gym` upstream'inin deployment-bridge checkpoint'leri (137k ve 164k).

**Bu arşiv yorum içermez.** Burada sadece veriler ve verinin ne olduğunu
tanımlayan bilgi vardır (format, şema, provenance, toplama protokolü). Kanaat,
yorum ve sentez raporları bilinçli olarak dışarıda bırakılmıştır — dışarıda
bırakılanların listesi ve orijinal konumları altta [Hariç tutulanlar](#hariç-tutulanlar)
bölümünde.

---

## Veri seti envanteri

| Klasör | Veri seti | Format | Örnek sayısı | Kaynak |
|---|---|---|---|---|
| `01_gate_probe/bizim_genesis_model_7500/` | Gate/routing bankası (Genesis, model_7500) | `.pt` banka + `.json` metrikler | 35.328 | `logs/eval/moe_gate_probe/Aug03_12-01-45_moe_cts_genesis/model_7500` |
| `01_gate_probe/upstream_mujoco/` | Upstream 137k + 164k MuJoCo closed-loop probe'ları | `.npz` + `.json` | 18.075 / checkpoint | `logs/eval/upstream_moe_cts` |
| `02_latent_probing/phaseA_23500_ve_best_tracking/` | Latent probing faz A: kontrollü fizik gridi (2 checkpoint) | `.npz` + `.json` | 92.160 / banka | `logs/eval/moe_latent_probe_phaseA` |
| `02_latent_probing/phaseB_checkpoint_trendi/` | Latent probing faz B: aynı protokol, 4 checkpoint trendi | `.npz` + `.json` | 92.160 / banka | `logs/eval/moe_latent_probe_phaseB` |
| `02_latent_probing/paper_pca/` + `paper_pca.zip` | PCA görsellerinin verisi (command/terrain) | `.npz` + `.json` + `.png` | — | `logs/eval/moe_paper_pca` |
| `03_brake/` | Fren (sim-to-sim) zaman serileri | `.npz` | 149 adım × 144 koşul | `logs/eval/brake` |
| `04_checkpoint_ve_tb/` | Eğitim ağırlıkları, TB event'leri, skaler dökümü | `.pt`, `.tfevents`, `.json` | — | `logs/go2_moects/*`, `logs/tb_merged/*`, `tmp/` |
| `05_veri_aciklamalari/` | Toplama yordamı, provenance kaydı, parametre envanteri | `.md`, `.py` | — | `scripts/`, `wiki/` |

---

## 01_gate_probe — gate/routing ölçüm bankaları

**Soru:** 8 uzmanın gate'i hangi girdilere göre routing yapıyor (veri düzeyinde).

### `bizim_genesis_model_7500/` — Genesis bankası

Kaynak: `logs/eval/moe_gate_probe/Aug03_12-01-45_moe_cts_genesis/model_7500`
Toplayan script: `legged_gym/scripts/eval/probe_moe_gate.py`

| Dosya | İçerik |
|---|---|
| `samples.pt` | Ana banka, dict of tensors (N = 35.328): `obs [N,45]`, `obs_history [N,225]`, `privileged_obs [N,263]`, `commands [N,3]`, `base_lin_vel [N,3]`, `base_ang_vel [N,3]`, `terrain_id [N]`, `terrain_level [N]` |
| `meta.json` | Provenance: checkpoint `model_7500` (iter 7500), `git_commit d138a2de` (dirty), seed 1, env config (384 env, DR: friction 0.5–1.5, added mass ±1 kg, COM, push), terrain config (heightfield `moe_grid`, 10×20), toplama ayarları |
| `results.json` | Analiz çıktısı (metrik değerleri): M1 per-expert norm, M2 uzmanlar arası cos (fonksiyonel fark), M3 gate entropisi/dağılımı, M4 ablasyon (learned/uniform), M5 terrain–uzman eşleşme (MI, chi2, contingency), M6 terrain sınıflandırıcı (MLP 225→256→128→9, 80/20 split) — her metrik kendi tanımıyla birlikte |
| `m1_effective_routing.png`, `m4_gate_ablation.png` | Metrik görselleri |
| `collect.log` | Toplama koşusu kütüğü |

Toplama protokolü (meta.json `collection`): warmup 100 adım, her 5 adımda örnek,
hedef 35.000 örnek, terrain başına en az 1.000, `recorded_steps` 92, durma nedeni
kayıtlı. 9 terrain etkin: wave, slope, rough_slope, stairs_up, stairs_down,
obstacles, stepping_stones, gap, flat. Sanity: `samples.pt` içindeki `g*E`
ağırlıklı latent ile forward latent farkı 0 (PASS).

### `upstream_mujoco/` — upstream checkpoint'leri, MuJoCo hattı

Kaynak: `logs/eval/upstream_moe_cts`
Toplayan script: `legged_gym/scripts/eval/probe_upstream_moe_cts.py`
İki checkpoint: `wty_go2_moe_cts_137k/` ve `wty_go2_moe_cts_high_slope_thre_164k/`.
Her birinin altında:

| Dosya | İçerik |
|---|---|
| `jit_parity/metrics.json` | Adapter↔JIT parity kanıtı: checkpoint SHA256'ları, `max_abs_action_error`/`mean_abs_action_error`, tolerans, `status: PASS`. Not: deployment-bridge şeması, `teacher_available=false` (yalnız actor + student MoE) |
| `closed_loop_exact_flat_stairs/metrics.json` | Koşu provenance'ı: checkpoint yolu, `deployment_bridge: true`, `critic_available: false`, dof_permutation, rollout koşulları |
| `closed_loop_exact_flat_stairs/probe.npz` | Kapalı-döngü bankası (N = 18.075): `obs [N,45]`, `history [N,5,45]`, `gate [N,8]`, `expert_outputs [N,8,32]`, `raw_weighted_latent / normalized_mixed_latent [N,32]`, `learned_action [N,12]`, `single_expert_action [N,8,12]`, `route_action [N,12]`, `commands`, `achieved_command_velocity`, `tracking_error`, `terrain_id/level/name`, `route_mode`, zaman/uzaklık bilgileri, karşı-olgusal bloklar: `gate_uniform`/`raw_latent_uniform`/`action_uniform`, `gate_shuffled`/..., `gate_top1`/..., `action_fixed_expert_0..7`, `current_obs`, `metadata_json` |

---

## 02_latent_probing — latent ne kodluyor veri bankaları

Toplayan script: `legged_gym/scripts/eval/probe_moe_latent.py`
Analiz eden script: `legged_gym/scripts/eval/probe_moe_analyze.py`
Tablo üreten script: `legged_gym/scripts/eval/probe_moe_report_tables.py`

**Grid protokolü (meta.json `mode: "grid"`):** her koşuda tek fizik ekseni
süpürülür, diğer beşi sabit. Eksenler: `friction`, `added_mass`, `com_x`,
`com_y`, `com_z`, `pd_gain_scale`. 6 komut × 12 değer × 8 env = 92.160 satır.
`axis_code [N]` (0–5) hangi eksenin süpürüldüğünü, `physics_raw/norm [N,6]`
eksen değerlerini taşır.

### `phaseA_23500_ve_best_tracking/`

| Dosya | İçerik |
|---|---|
| `best_tracking/{meta.json, samples.npz}` | Grid bankası — `best_tracking` checkpoint'i |
| `model_23500/{meta.json, samples.npz}` | Grid bankası — `model_23500` checkpoint'i |
| `intervene_friction_hi/intervene.npz` | Nedensel müdahale bankası, 51.200 satır |
| `intervene_friction_lo/intervene.npz` | Nedensel müdahale bankası, 51.200 satır |
| `intervene_pdgain_hi/intervene.npz` | Nedensel müdahale bankası, 51.200 satır |
| `analysis_fixed/{best_tracking, model_23500}/{probe_metrics.json, delta_r2_heatmap.png}` | Geçerli analiz çıktıları (düzeltilmiş fold ayrımı hattı) |
| `analysis_fixed/checkpoint_trend.png` | Checkpoint trendi görseli |

`samples.npz` şeması (N = 92.160): `obs [N,45]`, `obs_history [N,225]`,
`priv_obs [N,263]`, `z_s / z_t / z_random [N,32]` (student latent, teacher latent,
rastgele-init encoder latent), `g [N,8]` (gate), `E_norms [N,8]`,
`student_action / teacher_action [N,12]`, `physics_raw / physics_norm [N,6]`,
`axis_code [N]`, `val_index [N]` (0–11, fold indeksi), `physics_combo_id [N]`,
`ood_flag [N]` (bool), `env_id / episode_id / step / done / fall / time_out [N]`,
`terrain_id / terrain_level [N]`, `command_id [N]`, `command [N,3]`,
`base_lin_vel / base_ang_vel [N,3]`, `contact_pattern [N]`,
`tracking_lin_err / tracking_ang_err [N]`, `gait_phase [N]`,
`torque_norm / dof_acc_norm [N]`, `command_change_time [N]`,
`ckpt_tag / ckpt_iter [N]`.

`intervene.npz` şeması (N = 51.200): `mode [N]` ∈ {`student`, `teacher_true`,
`shuffled_matched`, `wrong_regime`}, `step [N]` (0–199), `pool_tier [N]` ∈
{-1, 0, 1, 2}, `env_id / episode_id / done / fall [N]`,
`tracking_lin_err / tracking_ang_err / achieved_speed / action_mae [N]`.

⚠️ **Not:** `intervene_pdgain_hi` kaynak klasöründe `intervene.npz` yanında
`intervene/`, `intervene (2)/`, `intervene (3)/` adlı klasörler vardır — bunlar
aynı bankanın `.npy` parçalarına açılmış kopyalarıdır (aktarım artığı, içerik
birebir `intervene.npz` ile aynı anahtarları taşır). Arşive sadece kanonik
`intervene.npz` alınmıştır.

⚠️ **Not:** Kaynak klasördeki `analysis/` (düzeltme öncesi hat: bozuk fold
ayrımı + seyreltilmiş hedef) geçersizdir ve arşive alınmamıştır. Geçerli
sayıların tamamı `analysis_fixed/` içindedir.

### `phaseB_checkpoint_trendi/`

Aynı grid protokolü, 4 checkpoint: `model_500`, `model_2500`, `model_7500`,
`model_12500`. Her birinde `meta.json` + `samples.npz` (92.160 satır).
`analysis/model_*/{probe_metrics.json, delta_r2_heatmap.png}` — metrik değerleri
ve ısı haritaları; `analysis/checkpoint_trend.png` — 6 checkpoint'lik trend
görseli (faz A'nın 20.500 ve 23.500'ü ile birlikte).

`probe_metrics.json` şeması: `samples` (örnek kümesi tanımı), `n_samples`,
`n_id`, `n_ood`, `group_key`, `fast_mode`, `features`, `controls`, `intervention`.

### `paper_pca/` + `paper_pca.zip`

Kaynak: `logs/eval/moe_paper_pca` (zip: `logs/eval/moe_paper_pca.zip`)
Üreten script: `legged_gym/scripts/eval/plot_moe_latent_pca.py`
Makaledeki PCA görselinin yeniden üretim verisi: `command_pca.{json,png}`,
`terrain_pca.{json,png}`, ve her checkpoint için `paper_pca/{meta.json, samples.npz}`
(`best_tracking/`, `model_23500/`).

---

## 03_brake — sim-to-sim fren zaman serileri

Kaynak: `logs/eval/brake`
Toplayan script: `legged_gym/scripts/eval/brake_probe.py` (+ `run_brake_probe.sh`)
Analiz eden script: `legged_gym/scripts/eval/brake_analyze.py`

| Dosya | İçerik |
|---|---|
| `genesis_student.npz`, `genesis_teacher.npz`, `genesis_selfreplay.npz` | Genesis tarafı rolları |
| `mujoco_student.npz`, `mujoco_teacher.npz`, `mujoco_selfreplay.npz` | MuJoCo tarafı rolları |
| `genesis_replay_mujoco_{sync,trig}.npz` | Genesis→MuJoCo açık-döngü replay merdiveni |
| `mujoco_replay_genesis_{sync,trig}.npz` | MuJoCo→Genesis açık-döngü replay merdiveni |
| `campaign.log`, `replay.log` | Toplama ve replay koşularının kütükleri |

Her `.npz` şeması: zaman serileri `[T, n_cond, ...]` ile T = 149 (dt = 0.02 s,
pre_steps 30, post_steps 100, warmup 150): `base_pos [149,144,3]`,
`base_quat [149,144,4]`, `base_lin_vel / base_ang_vel [149,144,3]`,
`projected_gravity [149,144,3]`, `dof_pos / dof_vel / torques / actions [149,144,12]`,
`feet_pos / feet_contact_forces [149,144,4,3]`, `commands [149,144,3]`,
`fell [149,144]`; meta skalerler: `backend`, `task`, `load_run`, `ckpt_path`
(model_20500), `arm`, `dt`, `T`, `pre_steps`, `post_steps`, `warmup`, `seed`,
`feet_names`, `dof_names`, `torque_limits`; koşul vektörleri:
`cond_vx [144]` (3 hız), `cond_ramp [144]` (3 rampa: 0/0.3/0.6),
`cond_phase [144]` (gait fazı), `trigger_step [144]`, `command_vx [149,144]`.

---

## 04_checkpoint_ve_tb — ağırlıklar ve eğitim eğrileri

| Dosya | İçerik |
|---|---|
| `birincil_kosu_Aug03_12-01-45/` → `logs/go2_moects/Aug03_12-01-45_moe_cts_genesis` | Bütün kampanyaların dayandığı koşu: `model_*.pt` (500→23.500), `best_spnte.pt`, `best_tracking.pt`, `exported/*.pt` (JIT), `go2_moects.py` + `go2_moects_config.py` (koşuda kullanılan config kopyaları), `events.out.tfevents.*` |
| `tb_merged_0_to_23540/` → `logs/tb_merged/moe_cts_0_to_23540` | 0→23.540 iter birleşik TensorBoard kütüğü (tek `events.out.tfevents.*`) |
| `moects_7500_tb.json` → `tmp/moects_7500_tb.json` | `model_7500` checkpoint'i için çıkarılmış skaler dökümü (`Episode/rew_*`, `Episode/termination_*`, `Episode/terrain_level_*`, `Episode/cmd_*` vb.) |
| `upstream_bridge_137k/` → `logs/go2_moects/wty_go2_moe_cts_137k` | Upstream deployment-bridge ağırlıkları: `model_0.pt` + `exported/*.pt` |
| `upstream_bridge_164k/` → `logs/go2_moects/wty_go2_moe_cts_high_slope_thre_164k` | Upstream deployment-bridge ağırlıkları: `model_0.pt` + `exported/*.pt` |

⚠️ Upstream bridge checkpoint'leri **yalnız actor + student MoE içerir**;
teacher/critic rastgele initialization'dadır (`teacher_available=false`,
`critic_available=false`). Bu checkpoint'lerden teacher latent'i veya
privileged oracle üretilmemelidir.

---

## 05_veri_aciklamalari — veriyi tanımlayan dokümanlar

| Dosya | İçerik |
|---|---|
| `uhem_runbook.md` → `scripts/moects_uhem_runbook.md` | UHeM/Altay üzerinde toplama koşularının yordamı |
| `uhem_provenance.py` → `scripts/moects_uhem_provenance.py` | Koşuların provenance kaydını tutan script |
| `wiki_moe_cts_parametreleri.md` → `wiki/MoE-CTS Parametreleri.md` | Ölçülen sistemin koddan derlenmiş parametre envanteri (verinin hangi config ile üretildiğini tanımlar) |

---

## Hariç tutulanlar

Aşağıdakiler bilinçli olarak arşive alınmamıştır; **orijinalleri repodaki
yerlerinde durmaktadır**:

| Ne | Neden hariç | Orijinal konum |
|---|---|---|
| Kanaat/sentez raporları (6 adet) | Yorum içerir; bu arşiv veri-only'dir | `moe_uzmanlasma_bizim_vs_upstream.md`, `upstream_mujoco_moe_uzmanlasma_sonuclari.md`, `moe_cts_kurtarma_planı.md`, `logs/eval/MOE_LATENT_PROBING_RAPOR.md`, `legged_gym/scripts/eval/BRAKE_PROBE.md`, `legged_gym/simulator/Evet, encoder düzeyinde kalmak mantıkl.md` |
| Tüm `REPORT.md`'ler (gate probe, latent probing analysis klasörleri) | Yorumlu özet; metrik verisi zaten `*.json` olarak arşivde | `logs/eval/moe_gate_probe/.../REPORT.md`, `logs/eval/moe_latent_probe_phase{A,B}/analysis*/.../REPORT.md` |
| Probe scriptleri, implementasyon kodu, testler | Kod — veri değil; üreten script yolları yukarıda her bölümde verilmiştir | `legged_gym/scripts/eval/*`, `rsl_rl/*`, `scripts/*`, `tests/*` |
| Probe planlama notları | Plan — veri tanımı değil | `wiki/Source - Probe Plans.md` |
| `phaseA` kaynağındaki `analysis/` (düzeltme öncesi hat) | Geçersiz ölçüm hattı (bozuk fold ayrımı + seyreltilmiş hedef); geçerli hat `analysis_fixed/` | `logs/eval/moe_latent_probe_phaseA/analysis/` |
| `intervene_pdgain_hi` altındaki `intervene/`, `intervene (2)/`, `intervene (3)/` | `intervene.npz`'nin parçalanmış kopyaları (içerik aynı) | `logs/eval/moe_latent_probe_phaseA/intervene_pdgain_hi/` |

---

## Bakım

- Yeni veri üretildiğinde: çıktıyı orijinal konumuna yaz, buraya symlink ekle ve
  ilgili bölüme bir satır düş. **Kanaat raporlarını bu arşive ekleme.**
- Kırık link kontrolü:

```bash
find moe_cts_veri -xtype l
```
