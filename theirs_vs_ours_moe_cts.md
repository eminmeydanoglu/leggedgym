## Net sonuç

`X` olarak `/home/emin/code/online-estimation/genesis-wp/LeggedGym-Ex/go2_rl_gym` içindeki upstream `go2_moe_cts` implementasyonunu, `Y` olarak da mevcut LeggedGym-Ex çalışma ağacındaki `go2_moects` implementasyonunu aldım.

Sonuç şu:

- Observation boyutları, terrain aileleri, command curriculum, reward katsayıları, network boyutları, expert sayısı ve temel MoE-CTS loss sözleşmesi büyük ölçüde eşleşiyor.
- Y jednak bit-for-bit veya davranışsal olarak birebir upstream port değil.
- En önemli farklar:
  1. X Isaac Gym + PhysX + trimesh, Y Genesis + Newton + heightfield kullanıyor.
  2. X base-contact termination eşiği `1.0 N`, Y `2.5 N`.
  3. X storage’a geçmeden önce role’leri teacher-first yeniden paketliyor; Y storage’ı fiziksel env sırasını koruyup role index’leri sonradan uyguluyor.
  4. Critic minibatch alignment bir fark **değildir**: X de Y de role başına tek index stream kullanıp critic alanlarını actor/history örnekleriyle aynı transition’da tutar. (Bağımsız all-env critic permutation’ı yalnızca Y’nin MoE yolunda kullanılmayan generic `RolloutStorageCTS`’inde vardır.)
  5. X başlangıç XY spawn jitter’ı `±1.0 m`, Y `±0.5 m`.
  6. X seed `0`, Y seed `1`.
  7. X motor mass/zero-offset/strength/restitution randomization’larını yapıyor; Y bunların önemli kısmını uygulamıyor.
  8. Y action-delay, control-rate DOF acceleration, Genesis friction ve collision ölçümünü adapte ediyor.
  9. Y terrain curriculum config’te `curriculum=False` olsa da `moe_grid=True` üzerinden curriculum’u mixin ile tekrar etkinleştiriyor.

Buradaki değerler “etkin config” değerleridir; yalnızca class constructor default’larını değil, inheritance sonrasında gerçekten kullanılan değerleri esas aldım. X tarafındaki `moe_cts.py` constructor default’ları bazı yerlerde farklı olsa da `GO2CfgMoECTS` sonunda PPO config değerlerini override ediyor.

### Kaynak dosya haritası

X:

- [X Go2 config](/home/emin/code/online-estimation/genesis-wp/LeggedGym-Ex/go2_rl_gym/legged_gym/envs/go2/go2_config.py:4)
- [X Go2 observation implementation](/home/emin/code/online-estimation/genesis-wp/LeggedGym-Ex/go2_rl_gym/legged_gym/envs/go2/go2_env.py:23)
- [X base environment](/home/emin/code/online-estimation/genesis-wp/LeggedGym-Ex/go2_rl_gym/legged_gym/envs/base/legged_robot.py:60)
- [X base config](/home/emin/code/online-estimation/genesis-wp/LeggedGym-Ex/go2_rl_gym/legged_gym/envs/base/legged_robot_config.py:4)
- [X MoE actor-critic](/home/emin/code/online-estimation/genesis-wp/LeggedGym-Ex/go2_rl_gym/rsl_rl/rsl_rl/modules/actor_critic_moe_cts.py:20)
- [X MoE algorithm](/home/emin/code/online-estimation/genesis-wp/LeggedGym-Ex/go2_rl_gym/rsl_rl/rsl_rl/algorithms/moe_cts.py:40)
- [X CTS storage](/home/emin/code/online-estimation/genesis-wp/LeggedGym-Ex/go2_rl_gym/rsl_rl/rsl_rl/storage/rollout_storage_cts.py:42)

Y:

- [Y MoE config](/home/emin/code/online-estimation/genesis-wp/LeggedGym-Ex/legged_gym/envs/go2/go2_moects/go2_moects_config.py:12)
- [Y MoE environment](/home/emin/code/online-estimation/genesis-wp/LeggedGym-Ex/legged_gym/envs/go2/go2_moects/go2_moects.py:14)
- [Y WTY curriculum/command/reset mixin](/home/emin/code/online-estimation/genesis-wp/LeggedGym-Ex/legged_gym/envs/go2/go2_moects/wty_curriculum_mixin.py:88)
- [Y MoE PPO](/home/emin/code/online-estimation/genesis-wp/LeggedGym-Ex/rsl_rl/algorithms/ppo_moe_cts.py:28)
- [Y MoE storage](/home/emin/code/online-estimation/genesis-wp/LeggedGym-Ex/rsl_rl/storage/rollout_storage_moe_cts.py:66)
- [Y MoE runner](/home/emin/code/online-estimation/genesis-wp/LeggedGym-Ex/rsl_rl/runners/moe_cts_runner.py:38)
- [Y Genesis simulator](/home/emin/code/online-estimation/genesis-wp/LeggedGym-Ex/legged_gym/simulator/genesis_simulator.py:48)
- [Y terrain generator](/home/emin/code/online-estimation/genesis-wp/LeggedGym-Ex/legged_gym/utils/terrain.py:596)

---

## 0. Kimlik, simulator ve karşılaştırılabilirlik

| Öncelik / grup | Parametre | X — `go2_rl_gym` | Y — LeggedGym-Ex |
|---|---|---|---|
| P0 / Kimlik | Task adı | `go2_moe_cts` | `go2_moects` |
| P0 / Kimlik | Environment class | `Go2Robot` | `Go2MoECTS` |
| P0 / Kimlik | Config class | `GO2Cfg + GO2CfgMoECTS` | `Go2MoECTSCfg + Go2MoECTSCfgPPO` |
| P0 / Snapshot | İncelenen source commit | `30e74dc` | `6d6058c` commit tabanı + dirty working tree |
| P0 / Snapshot | Çalışma ağacı | Temiz upstream checkout | Modifiye edilmiş; config, env, simulator, PPO, storage ve test dosyalarında yerel değişiklikler var |
| P0 / Simulator | Fizik motoru | Isaac Gym / PhysX | Genesis / Newton constraint solver |
| P0 / Terrain | Terrain mesh türü | Effective değer: `trimesh` | Effective değer: `heightfield` |
| P0 / Terrain | Terrain oluşturma | Isaac Gym `terrain_utils` → heightfield → trimesh dönüşümü | Genesis heightfield entity |
| P0 / Terrain | Y’nin X ile fiziksel eşdeğerliği | PhysX temas, friction combine ve collision çözümü | Newton temas çözümü; doğrudan fiziksel eşdeğer değil |
| P0 / Fairness | Aynı config ile aynı dünya mı? | Hayır | Hayır; numeric config eşleşse bile simulator ve terrain representation farklı |
| P0 / Rate | Simulator timestep | `0.005 s` | `0.005 s` |
| P0 / Rate | Simulator substeps | `1` | `1` |
| P0 / Rate | Control timestep | `0.02 s` | `0.02 s` |
| P0 / Rate | Control frequency | `50 Hz` | `50 Hz` |
| P0 / Control | Decimation | `4` | `4` |
| P0 / Control | Physics steps / policy action | 4 | 4 |
| P1 / Gravity | Gravity | `[0, 0, -9.81]` | `[0, 0, -9.81]` |
| P1 / Gravity | Up axis | Z | Z |
| P1 / Solver | Collision solver | PhysX TGS, `solver_type=1` | Genesis Newton |
| P1 / Solver | Position iterations | `4` | Doğrudan eşdeğeri yok |
| P1 / Solver | Velocity iterations | `0` | Doğrudan eşdeğeri yok |
| P1 / Solver | Max collision pairs | PhysX `2**23` | Genesis `100` |
| P1 / Solver | IK max targets | X config’te yok | `2` |
| P2 / Plane | `env_spacing` | `3.0 m`, terrain mesh’te kullanılmıyor | `2.0 m`, heightfield’te kullanılmıyor |
| P2 / Terrain support | Genesis’te trimesh | Uygulanabilir | Y simulator path’i trimesh için `NotImplementedError` veriyor |
| P2 / Terrain support | Heightfield edge handling | Isaac Gym conversion path’i | Explicit heightfield ve edge-mask path’i |

Bu nedenle eğitim sonucunu “X’in Genesis portu aynı davranıyor” şeklinde yorumlamak için yalnızca config diff’i yeterli değil. Özellikle termination, friction ve collision kaynaklı farklar simulator seviyesinde ölçülmeli.

---

## 1. Observation, privileged observation ve tensor ABI

| Öncelik / grup | Parametre | X — `go2_rl_gym` | Y — LeggedGym-Ex |
|---|---|---|---|
| P0 / Boyut | Parallel env sayısı | `8192` | `8192` |
| P0 / Boyut | Actor observation | `45` | `45` |
| P0 / Boyut | Privileged observation | `263` | `263` |
| P0 / Boyut | Action boyutu | `12` | `12` |
| P0 / Actor obs | İlk bölüm | `base_ang_vel`, 3 | Aynı |
| P0 / Actor obs | İkinci bölüm | `projected_gravity`, 3 | Aynı |
| P0 / Actor obs | Command bölümü | `commands[:3]`, 3 | Aynı |
| P0 / Actor obs | DOF position error | `dof_pos - default_dof_pos`, 12 | Aynı |
| P0 / Actor obs | DOF velocity | `dof_vel`, 12 | Aynı |
| P0 / Actor obs | Previous/current action feature | `actions`, 12 | Aynı |
| P0 / Actor obs | Toplam | `3 + 3 + 3 + 12 + 12 + 12 = 45` | Aynı |
| P0 / Actor obs | Sıralama | `[ang_vel, gravity, commands, dof_pos_error, dof_vel, actions]` | Aynı |
| P0 / Actor obs | Observation noise | Actor observation’a ekleniyor | Actor observation’a ekleniyor |
| P0 / Actor obs | Commands noise | Yok | Yok |
| P0 / Actor obs | Action noise | Yok | Yok |
| P0 / Privileged | Base linear velocity | 3 | 3 |
| P0 / Privileged | Temiz actor observation | Temiz, noise eklenmeden önceki 45-D observation | Temiz, noise eklenmeden önceki 45-D observation |
| P0 / Privileged | Foot contact force | 4 | 4 |
| P0 / Privileged | Normalized motor torque | `torques / torque_limits`, 12 | Aynı |
| P0 / Privileged | DOF acceleration | 12 | Aynı formül, fakat Y control-rate tracker kullanıyor |
| P0 / Privileged | Height scan | 187 | 187 |
| P0 / Privileged | Toplam | `3 + 45 + 4 + 12 + 12 + 187 = 263` | Aynı |
| P0 / Height scan | Height ölçüm noktası | `17 × 11 = 187` | `17 × 11 = 187` |
| P0 / Height scan | X aralığı | `[-0.8, ..., 0.8]`, 17 nokta | Aynı |
| P0 / Height scan | Y aralığı | `[-0.5, ..., 0.5]`, 11 nokta | Aynı |
| P0 / Height scan | Height preprocessing | `root_z - 0.5 - measured_height`, clip `[-1,1]`, scale `2.5` | Aynı |
| P0 / History | History frame sayısı | `5` | `5` |
| P0 / History | History boyutu | `5 × 45 = 225` | `225` |
| P0 / History | History sahibi | X runner/model tarafında tutuluyor; inference’ta modelin internal history buffer’ı var | Y environment deque’si; eval’de `HistoryObsAdapter` |
| P0 / History | History içeriği | Gürültülü 45-D actor obs frame’leri | Gürültülü 45-D actor obs frame’leri |
| P0 / History | Done sonrası history | Done env history sıfırlanıyor | Host/Y history lifecycle ile sıfırlanıyor |
| P0 / Critic | Critic observation frame stack | `1` | `1` |
| P0 / Critic | Critic observation boyutu | `263` | `263` |
| P0 / Critic | Critic input | `[latent, privileged_obs]` | `[latent, critic_obs]` |
| P0 / Critic | Critic input boyutu | `32 + 263 = 295` | `32 + 263 = 295` |
| P0 / Actor | Actor input | `[latent, actor_obs]` | `[latent, actor_obs]` |
| P0 / Actor | Actor input boyutu | `32 + 45 = 77` | `77` |
| P0 / Latent | Latent boyutu | `32` | `32` |
| P0 / Latent | Latent normalization | `l2norm` | `l2norm` |
| P0 / Network | Activation | `elu` | `elu` |
| P0 / Network | Actor hidden layers | `[512, 256, 128]` | `[512, 256, 128]` |
| P0 / Network | Critic hidden layers | `[512, 256, 128]` | `[512, 256, 128]` |
| P0 / Network | Teacher / privilege encoder | `[512, 256] → 32` | `[512, 256] → 32` |
| P0 / Network | Student encoder | `[512, 256, 256] → 32` | `[512, 256, 256] → 32` |
| P0 / MoE | Expert sayısı | `8` | `8` |
| P0 / MoE | Gating | Softmax expert weights | Softmax expert weights |
| P0 / MoE | Expert usage telemetry | Kaynakta doğrudan sınırlı | Y `entropy`, effective experts, min/max/std ve usage listesi logluyor |
| P0 / Action distribution | Initial action std | `1.0` | `1.0` |
| P0 / Gradient | Student encoder PPO gradient’i | Student action forward `no_grad`; student encoder PPO’dan güncellenmiyor | Aynı; student encoder PPO `rl_params` dışında |
| P0 / Gradient | Critic → encoder gradient’i | Critic concat öncesi latent detach | Aynı |
| P0 / Distillation | Latent target | Teacher latent | Privilege encoder latent |
| P0 / Distillation | Target gradient’i | `stopgrad(teacher_latent)` | `torch.no_grad()` |
| P0 / Distillation | Student loss | `MSE(student, teacher)` | Aynı |
| P0 / Load balance | Load-balance loss | Mean expert usage’ı `1/8` uniform dağılıma yaklaştırıyor | Aynı |
| P0 / Load balance | Katsayı | `0.01` | `0.01` |
| P1 / DOF order | Action/observation DOF sırası | URDF asset sırasından geliyor; config’te açık bir `dof_names` listesi yok | Açık sıra: `FR`, `FL`, `RR`, `RL`; her bacakta hip/thigh/calf |
| P1 / Critic | Teacher critic latent | Privileged encoder latent’i | Privilege encoder latent’i |
| P1 / Critic | Student critic latent | Student MoE history latent’i | Student MoE history latent’i |
| P1 / Critic | Critic latent gradient | Detached | Detached |

Observation sözleşmesinde Y’nin X’e en yakın kısmı burası. `45/263/225/295` boyutlarının eşleşmesi gerçek bir parity işareti; fakat bu, storage ve PPO update farklarının da aynı olduğu anlamına gelmiyor.

---

## 2. Eğitim ve PPO parametreleri

| Öncelik / grup | Parametre | X — `go2_rl_gym` | Y — LeggedGym-Ex |
|---|---|---|---|
| P0 / Training | Random seed | `0` | `1` |
| P0 / Training | Episode length | `25 s` | `25 s` |
| P0 / Training | Control steps / episode | `25 / 0.02 = 1250` | `1250` |
| P0 / Rollout | Steps per env per iteration | `24` | `24` |
| P0 / Rollout | Total rollout transitions | `8192 × 24 = 196608` | Aynı |
| P0 / Role | Teacher ratio | `0.75` | `0.75` |
| P0 / Role | Teacher env sayısı | `int(8192 × .75) = 6144` | `6144` |
| P0 / Role | Student env sayısı | `2048` | `2048` |
| P0 / Role | Student oranı | `0.25` | `0.25` |
| P0 / Role | Student mapping | Her 4. env student: `0,4,8,...` | Aynı interleaved mapping |
| P0 / PPO | Value loss coefficient | `1.0` | `1.0` |
| P0 / PPO | Clipped value loss | Açık | Açık |
| P0 / PPO | PPO clip parameter | `0.2` | `0.2` |
| P0 / PPO | Entropy coefficient | Effective config: `0.01` | `0.01` |
| P0 / PPO | Discount gamma | Effective config: `0.99` | `0.99` |
| P0 / PPO | GAE lambda | `0.95` | `0.95` |
| P0 / PPO | Learning rate | `1e-3` | `1e-3` |
| P0 / PPO | Student encoder learning rate | `1e-3` | `1e-3` |
| P0 / PPO | Learning-rate schedule | `adaptive` | `adaptive` |
| P0 / PPO | Desired KL | `0.01` | `0.01` |
| P0 / PPO | Max grad norm | `1.0` | `1.0` |
| P0 / PPO | Learning epochs | `5` | `5` |
| P0 / PPO | Mini-batches | `4` | `4` |
| P0 / PPO | PPO gradient steps / rollout | `5 × 4 = 20` | `20` |
| P0 / Encoder | Encoder epochs | Kaynakta ikinci pass aynı minibatch listesi üzerinde `1` effective encoder epoch | `1` |
| P0 / Encoder | Encoder optimizer steps | `20` student update | `20` student update |
| P0 / MoE | Load balance coefficient | `0.01` | `0.01` |
| P0 / Training | Max iterations | `150000` | `150000` |
| P0 / Checkpoint | Save interval | `500` | `500` |
| P1 / PPO batch | Teacher samples / minibatch | `6144 × 24 / 4 = 36864` | `36864` |
| P1 / PPO batch | Student samples / minibatch | `2048 × 24 / 4 = 12288` | `12288` |
| P1 / PPO batch | Total policy samples / minibatch | `49152` | `49152` |
| P1 / CLI | `--num_envs` değişince role sayısı | Algorithm runtime’da yeniden hesaplıyor | Environment mixin `num_teacher`’ı yeniden ölçekliyor; algorithm aynı değeri assert ediyor |
| P1 / Runner | Runner class | `OnPolicyRunnerCTS` | `MoECTSRunner` |
| P1 / Runner | Algorithm class | `MoECTS` | `PPO_MOE_CTS` |
| P1 / Runner | Actor class | `ActorCriticMoECTS` | `ActorCriticMoECTS` |
| P2 / Resume | Default resume | `False` | Host runner sözleşmesi; default false |
| P2 / Resume | Last run | `-1` | `-1` |
| P2 / Resume | Last checkpoint | `-1` | `-1` |
| P2 / Logging | X run identity | Experiment `go2_moe_cts` | Experiment `go2_moects`, Genesis run name `moe_cts_genesis` |
| P2 / Logging | Per-role reward logging | Teacher/student ayrı kanallar var | Aggregate reward yanında teacher/student ayrı kanallar var |
| P2 / Logging | MoE telemetry | Temel loss bilgileri | Latent MSE, load balance, gating entropy, effective expert, usage istatistikleri |

### Constructor default notu

X’in `moe_cts.py` fonksiyon imzasında bazı default’lar `gamma=0.998`, `entropy_coef=0.0`, `schedule="fixed"`, `epochs=1`, `minibatches=1` olarak duruyor. Fakat task config’i `LeggedRobotCfgCTS` üzerinden bunları etkin olarak şu değerlere getiriyor:

- `gamma=0.99`
- `entropy_coef=0.01`
- `schedule="adaptive"`
- `num_learning_epochs=5`
- `num_mini_batches=4`

Tabloda constructor default’larını değil, task’ın effective değerlerini kullandım.

---

## 3. Teacher/student role mapping, storage ve PPO update farkları

Bu bölüm en kritik bölümlerden biri. Buradaki farklar network boyutlarından daha önemli olabilir.

| Öncelik / grup | Parametre | X — `go2_rl_gym` | Y — LeggedGym-Ex |
|---|---|---|---|
| P0 / Role mapping | Fiziksel env role dağılımı | Interleaved: student env’ler `0,4,8,...`; diğerleri teacher | Aynı |
| P0 / Forward | Teacher forward | `obs[teacher_ids]`, privileged obs ile | Aynı gather yaklaşımı |
| P0 / Forward | Student forward | `obs[student_ids]`, history ile | Aynı |
| P0 / Action | Env’e dönen action sırası | Role-grouped sonuçlar tekrar fiziksel env sırasına scatter ediliyor | Env sırası korunuyor, role sonuçları env sırasına scatter ediliyor |
| P0 / Storage | Storage’daki env sırası | Teacher-first / student-second olarak yeniden paketlenmiş | Orijinal env sırası korunuyor |
| P0 / Storage | Transition reward/done sırası | `rewards[teacher_ids]` sonra `rewards[student_ids]` | Reward/done env sırasıyla yazılıyor |
| P0 / Storage | Teacher/student storage bölgesi | İlk `6144` sütun teacher, son `2048` student | Fiziksel interleaved id’ler üzerinden gather |
| P0 / GAE | Teacher GAE | Teacher-first slice üzerinde | Teacher env index’leri üzerinde |
| P0 / GAE | Student GAE | Student slice üzerinde | Student env index’leri üzerinde |
| P0 / GAE | GAE recurrence | Aynı `gamma=0.99`, `lambda=.95` | Aynı |
| P0 / Advantage | Advantage normalization | Teacher + student tek storage buffer’ında global mean/std ile normalize ediliyor | Teacher ve student advantage’ları concat edildikten sonra tek global mean/std ile normalize ediliyor |
| P0 / Advantage | Normalizasyon parity’si | `self.advantages.mean()` / `self.advantages.std()` combined buffer üzerinde | `all_advantages.mean()` / `all_advantages.std()` combined teacher+student üzerinde |
| P0 / Critic minibatch | Critic sample seçimi | Role başına tek index stream (`teacher_indices`, `student_indices`); `critic_obs`/`values`/`returns` actor alanlarıyla aynı slice’tan geliyor ([rollout_storage_cts.py:158](/home/emin/code/online-estimation/genesis-wp/LeggedGym-Ex/go2_rl_gym/rsl_rl/rsl_rl/storage/rollout_storage_cts.py:158)) | Aynı sözleşme: role başına tek index stream, critic alanları aynı slice’tan |
| P0 / Critic minibatch | History–critic eşleşmesi | Aynı transition; `evaluate(privileged_obs[start:end], history[start:end])` `act` ile aynı dilim ([moe_cts.py:122](/home/emin/code/online-estimation/genesis-wp/LeggedGym-Ex/go2_rl_gym/rsl_rl/rsl_rl/algorithms/moe_cts.py:122)) | Aynı transition’a ait `critic_obs`, history, value target ve return birlikte kalıyor |
| P0 / Critic minibatch | Critic role purity | Role-pure: teacher yarısı teacher index’lerinden, student yarısı student index’lerinden; loss’ta concat ediliyor | Aynı; ayrıca iki ayrı `CriticMiniBatch` olarak yield ediliyor |
| P0 / Critic minibatch | Yapısal fark | Tek 12-tuple, teacher-first concat; role’ler `[:teacher_samples]` / `[teacher_samples:]` slice’ıyla ayrılıyor | Ayrı `teacher_batch`/`student_batch`/`teacher_critic`/`student_critic` tuple’ları; içerik aynı, yalnızca paketleme farklı |
| P0 / Value loss | Teacher/student value loss aggregation | Concatenated teacher+student per-sample value losses üzerinde tek `mean()` | Concatenated teacher+student per-sample value losses üzerinde tek `mean()` |
| P0 / Surrogate | Teacher surrogate | PPO clipped surrogate | Aynı |
| P0 / Surrogate | Student surrogate | PPO clipped surrogate | Aynı |
| P0 / Surrogate | Teacher/student surrogate birleşimi | `teacher_loss + student_loss` | `teacher_loss + student_loss` |
| P0 / Entropy | Entropy aggregation | Teacher/student entropy concat mean | Aynı |
| P0 / KL | Adaptive KL | Teacher/student yeniden hesaplanan distribution’lar concat ediliyor | Teacher ve student old/new mu-sigma concat ediliyor |
| P0 / Optimizer 1 | Güncellenen parametreler | `teacher_encoder`, `actor`, `critic`, `std` | `privilege_encoder`, `actor`, `critic`, `std` |
| P0 / Optimizer 2 | Güncellenen parametreler | `student_moe_encoder` | `history_encoder` |
| P0 / Gradient isolation | Student → PPO | Student actor forward `no_grad` | Aynı |
| P0 / Gradient isolation | Critic → encoder | Latent detach | Aynı |
| P0 / Encoder pass | Teacher target | Teacher encoder output, no gradient | Privilege encoder output, no gradient |
| P0 / Encoder pass | Student input | Student history batch | Student history batch |
| P0 / Encoder loss | Formula | `MSE(student_latent, stopgrad(teacher_latent)) + .01 × load_balance` | Aynı |
| P1 / Minibatch permutation | Epoch permütasyonu | Storage generator permutation’ı epoch loop dışında üretiyor; kaynak aynı listeyi tekrar kullanıyor | Aynı davranış varsayımı explicit `minibatches_are_epoch_invariant` ile kullanılıyor |
| P1 / Bellek | Rollout materialization | Kaynak tüm epoch listesini materialize ediyor | Y bir epoch materialize edip 5 kez replay ediyor; gradient step sayısı aynı |
| P1 / Storage validation | Role mapping validation | Contiguous storage sözleşmesine dayanıyor; tutarlı, çünkü `act()` interleaved role’leri storage’a yazmadan önce teacher-first paketliyor ([cts.py:136-141](/home/emin/code/online-estimation/genesis-wp/LeggedGym-Ex/go2_rl_gym/rsl_rl/rsl_rl/algorithms/cts.py:136)) | Interleaved index’ler explicit ve host contiguous CTS fallback’i reddediliyor |
| P1 / Bootstrap | Last critic value | Teacher-first concat sonra storage’a veriliyor | Env-order teacher/student values scatter edilip storage’a veriliyor |
| P1 / Timeout bootstrap | Timeout reward correction | `gamma × value × timeout` ekleniyor | Aynı sözleşme |
| P2 / Storage ABI | Transition alanları | `history`, `critic_observations`, `privileged_observations` kaynak alan adları | `observation_histories`, `critic_observations`, env-order alanları |
| P2 / Model ABI | Encoder isimleri | `teacher_encoder`, `student_moe_encoder` | `privilege_encoder`, `history_encoder` |
| P2 / Import | Upstream checkpoint doğrudan yükleme | Kaynak model isimleri | Y import/deploy path’i isim mapping’i gerektiriyor |

Burada önemli bir düzeltme: MoE-CTS advantage normalizasyonu bir fark değildir. X global combined buffer üzerinde, Y ise teacher/student advantage’larını concat ettikten sonra global olarak normalize eder. Y’de GAE’nin iki role loop’unda koşması da env başına bağımsız recurrence nedeniyle matematiksel bir davranış farkı oluşturmaz.

Kapsam notu: Y’deki generic/non-MoE [`RolloutStorageCTS`](/home/emin/code/online-estimation/genesis-wp/LeggedGym-Ex/rsl_rl/storage/rollout_storage_cts.py:95) teacher ve student advantage’larını ayrı normalize eder. Ancak MoE-CTS PPO yolu bunu kullanmaz; [`RolloutStorageMoECTS`](/home/emin/code/online-estimation/genesis-wp/LeggedGym-Ex/rsl_rl/storage/rollout_storage_moe_cts.py:151) kullanır ve teacher+student üzerinde global `mean/std` uygular.

İkinci düzeltme (bu dokümanın önceki sürümünde X’e yanlış atfedilmişti): **critic minibatch alignment da bir fark değildir.** X’in MoE-CTS storage’ı yalnızca iki index stream üretir ([rollout_storage_cts.py:158-159](/home/emin/code/online-estimation/genesis-wp/LeggedGym-Ex/go2_rl_gym/rsl_rl/rsl_rl/storage/rollout_storage_cts.py:158)) ve `get_teacher_student_samples` aynı slice’ı `observations`, `critic_observations`, `history`, `values`, `returns`, `mu/sigma` dahil her alana uygular; `moe_cts.py:122-124`’te `act` ve `evaluate` aynı `[start:end]` dilimini kullanır. Bağımsız üçüncü bir permutation (`total_indices`) yalnızca Y’nin generic [`RolloutStorageCTS.mini_batch_generator`](/home/emin/code/online-estimation/genesis-wp/LeggedGym-Ex/rsl_rl/storage/rollout_storage_cts.py:123) yolundadır ve MoE-CTS bu yolu kullanmaz. Yani `RolloutStorageMoECTS`’in host CTS’ten ayrışması, X’ten sapma değil, X ile hizalanmadır — MoE için zorunludur da, çünkü student critic `evaluate(critic_obs, history)` alır ve host generator’da history critic batch’iyle hizasızdır.

Value loss aggregation’ı da fark değildir: iki tarafta da teacher+student per-sample value loss’ları tek combined mean ile toplanır.

Dolayısıyla PPO update semantics tarafında geriye kalan farklar yapısal/ABI düzeyindedir (tuple paketlemesi, storage’ın env sırası + role index’leri, encoder isimleri); sample contract’ı değiştiren bir fark kalmamıştır.

---

## 4. Terrain ve dünya parametreleri

| Öncelik / grup | Parametre | X — `go2_rl_gym` | Y — LeggedGym-Ex |
|---|---|---|---|
| P0 / Terrain | Terrain curriculum bankı | `10 × 20 = 200` tile | `10 × 20 = 200` tile |
| P0 / Terrain | Satır anlamı | Difficulty level | Difficulty level |
| P0 / Terrain | Sütun anlamı | Terrain family/type | Terrain family/type |
| P0 / Terrain | Tile length | `8.0 m` | `8.0 m` |
| P0 / Terrain | Tile width | `8.0 m` | `8.0 m` |
| P0 / Terrain | Terrain spacing | `0.5 m` | `0.5 m` |
| P0 / Terrain | Border | `25 m` | `25 m` |
| P0 / Resolution | Horizontal scale | `0.1 m` | `0.1 m` |
| P0 / Resolution | Vertical scale | `0.005 m` | `0.005 m` |
| P0 / Terrain | Terrain proportions | `[.05,.20,.05,.25,.10,.20,0,0,.15]` | Aynı |
| P0 / Terrain | Terrain family sırası | Wave, slope, rough slope, stairs up, stairs down, obstacles, stepping stones, gap, flat | Aynı |
| P0 / Terrain | Effective column dağılımı | Wave `1`, slope `4`, rough slope `1`, stairs up `5`, stairs down `2`, obstacles `4`, stepping `0`, gap `0`, flat `3` | Aynı |
| P0 / Terrain | Level difficulty | `i / num_rows`, yani `0.0 ... 0.9` | Aynı |
| P0 / Terrain | Terrain ID’leri | Wave `0`, slope `1`, rough slope `2`, stairs up `3`, stairs down `4`, obstacles `5`, stepping `6`, gap `7`, flat `8` | Aynı |
| P0 / Terrain | Terrain type bookkeeping | `name2cols`, `cols2id` | Aynı; Y ayrıca command/rank metric’leri için kullanıyor |
| P0 / Terrain | Terrain generation mode | `cfg.curriculum=True` ile `curiculum()` | `cfg.curriculum=False`, fakat `moe_grid=True` branch’i üzerinden `moe_grid()` |
| P0 / Terrain | Curriculum effective mi? | Evet | Evet; config flag false olsa da mixin explicit etkinleştiriyor |
| P0 / Terrain | Heightfield/trimesh | Önce heightfield üretip trimesh’e dönüştürüyor | Heightfield-only |
| P0 / Terrain | Map allocation | Spacing boşlukları toplam map boyutuna dahil | Y `_moe_alloc_map()` ile spacing’i explicit dahil ediyor |
| P0 / Terrain | Tile origin X | `(i + .5) × 8 + i × .5` | Aynı |
| P0 / Terrain | Tile origin Y | `(j + .5) × 8 + j × .5` | Aynı |
| P0 / Terrain | Origin Z | Merkezdeki yaklaşık `2 m × 2 m` patch’in maksimum yüksekliği | Aynı |
| P1 / Terrain | Max initial level | `5` | `5` |
| P1 / Terrain | Initial terrain level dağılımı | Env’ler `0 ... max_init_level` arasında round-robin | Aynı |
| P1 / Terrain | Initial terrain type dağılımı | Env ID’leri column’lara gruplanıyor | Aynı semantic column mapping |
| P1 / Terrain | Promotion ölçütü | Max XY move distance `> terrain_length / 2` | Aynı |
| P1 / Terrain | Demotion ölçütü | Accumulated XY command açıksa accumulated command distance ile | Aynı |
| P1 / Terrain | Demotion çarpanı | `resampling_time × (1-zero_command_prob) × 0.5` | Aynı |
| P1 / Terrain | Demotion alternatif kuralı | Accumulation kapalıysa current command norm × episode length × `.5` | Aynı |
| P1 / Terrain | Max level sonrası | Random level `[0, max_level)` | Aynı |
| P1 / Terrain | Min level | `0` | `0` |
| P1 / Terrain | Terrain curriculum update zamanı | Reset sırasında | Reset sırasında |
| P1 / Terrain | Terrain level metrikleri | Basic terrain level bilgileri | Y terrain-family bazlı episode telemetry de yazıyor |
| P1 / Geometry | `IS_HARD` | Açık | Açık |
| P1 / Geometry | Slope formülü | `0.1 + .52 × difficulty` | Aynı |
| P1 / Geometry | Maksimum slope | Yaklaşık `0.568 rad`, `29.6°` | Aynı |
| P1 / Geometry | Stair height | `.05 + .23 × difficulty` | Aynı |
| P1 / Geometry | Maksimum stair height | Yaklaşık `.257 m` | Aynı |
| P1 / Geometry | Obstacle height | `.05 + .25 × difficulty` | Aynı |
| P1 / Geometry | Maksimum obstacle height | Yaklaşık `.275 m` | Aynı |
| P1 / Geometry | Stepping stone size | `1.5 × (1.05 - difficulty)` | Aynı |
| P1 / Geometry | Stone distance | Difficulty `0` ise `.05`, aksi halde `.1` | Aynı |
| P1 / Geometry | Gap size | `difficulty` | Aynı |
| P1 / Geometry | Wave amplitude | `.1 + .2 × difficulty` | Aynı |
| P1 / Geometry | Wave count | `5` | `5` |
| P1 / Geometry | Roughness | Uniform noise `[-.05,.05]`, step `.005`, downsample `.2` | Aynı |
| P1 / Geometry | Slope platform | `3 m` | `3 m` |
| P1 / Geometry | Stair width | `.31 m` | `.31 m` |
| P1 / Geometry | Obstacle sayısı | `20` | `20` |
| P1 / Geometry | Obstacle boyutları | `1–2 m` | `1–2 m` |
| P1 / Geometry | Stepping platform | `4 m` | `4 m` |
| P1 / Geometry | Flat terrain | `pit_terrain(depth=0, platform_size=4)` | Aynı |
| P2 / Terrain | Terrain RNG | Dedicated terrain seed yok; genel RNG’ye bağlı | Dedicated terrain seed yok; genel RNG’ye bağlı |
| P2 / Terrain | Terrain reproducibility | Aynı seed ve aynı simulator olmadan map birebirliği garanti değil | Aynı |
| P2 / Terrain | `slope_threshold` | `1.5`; trimesh conversion için etkili | Y inherited yaklaşık `.75`; heightfield path’inde pratikte etkisiz |
| P2 / Terrain | `platform_size` config | Genel default `3`; family helper’ları bazı yerlerde `3/4` hardcode ediyor | Aynı helper davranışı |
| P2 / Terrain | Selected terrain | `False` | `False` |
| P2 / Terrain | Curriculum + selected çakışması | Guard var | Guard var; `moe_grid` branch’i ikisinden önce kazanıyor |
| P2 / Terrain | Edge mask | Kaynak terrain class’ında aynı explicit edge-mask sözleşmesi yok | Y heightfield edge mask tutuyor; Genesis terrain sınırı/reward telemetry için kullanılabilir |

Y’nin terrain geometry’si kaynak formüllerine oldukça yakın. Buradaki büyük semantik fark geometry değil, `trimesh/PhysX` ile `heightfield/Newton` farkı.

---

## 5. Command sistemi, sınırlar ve command curriculum

| Öncelik / grup | Parametre | X — `go2_rl_gym` | Y — LeggedGym-Ex |
|---|---|---|---|
| P0 / Command | Command boyutu | `4` | `4` |
| P0 / Command | Command anlamı | `lin_vel_x`, `lin_vel_y`, `ang_vel_yaw`, `heading` | Aynı |
| P0 / Command | Heading command | `False` | `False` |
| P0 / Command | Command resampling | `5 s` | `5 s` |
| P0 / Global bounds | Başlangıç `lin_vel_x` | `[-.5,.5] m/s` | Aynı |
| P0 / Global bounds | Başlangıç `lin_vel_y` | `[-.5,.5] m/s` | Aynı |
| P0 / Global bounds | Başlangıç yaw rate | `[-1,1] rad/s` | Aynı |
| P0 / Global bounds | Heading bound | `[-1.57,1.57]` | Aynı |
| P0 / Iteration curriculum | İlk command stage | Başlangıç global bounds | Aynı |
| P0 / Iteration curriculum | Stage 1 zamanı | `20000` iteration | `20000` |
| P0 / Iteration curriculum | Stage 1 X velocity | `[-1,1] m/s` | Aynı |
| P0 / Iteration curriculum | Stage 1 Y velocity | `[-1,1] m/s` | Aynı |
| P0 / Iteration curriculum | Stage 1 yaw | `[-1.5,1.5] rad/s` | Aynı |
| P0 / Iteration curriculum | Stage 1 heading | `[-1.57,1.57]` | Aynı |
| P0 / Iteration curriculum | Stage 2 zamanı | `50000` iteration | `50000` |
| P0 / Iteration curriculum | Stage 2 X velocity | `[-2,2] m/s` | Aynı |
| P0 / Iteration curriculum | Stage 2 Y velocity | `[-1,1] m/s` | Aynı |
| P0 / Iteration curriculum | Stage 2 yaw | `[-2,2] rad/s` | Aynı |
| P0 / Iteration curriculum | Stage 2 heading | `[-1.57,1.57]` | Aynı |
| P0 / Terrain bounds | Wave/slope/rough X | `[-1.5,1.5]` | Aynı |
| P0 / Terrain bounds | Wave/slope/rough Y | `[-1,1]` | Aynı |
| P0 / Terrain bounds | Wave/slope/rough yaw | `[-1.5,1.5]` | Aynı |
| P0 / Terrain bounds | Stairs/obstacle X | `[-1,1]` | Aynı |
| P0 / Terrain bounds | Stairs/obstacle Y | `[-1,1]` | Aynı |
| P0 / Terrain bounds | Stairs/obstacle yaw | `[-1.5,1.5]` | Aynı |
| P0 / Terrain bounds | Flat X | `[-2,2]` | Aynı |
| P0 / Terrain bounds | Flat Y | `[-1,1]` | Aynı |
| P0 / Terrain bounds | Flat yaw | `[-2,2]` | Aynı |
| P0 / Dynamic sample | Dynamic resampling | Açık | Açık |
| P0 / Dynamic sample | Remaining distance | `clip(.625 × terrain_length - ||xy_accum|| × resampling_time, 0)` | Aynı |
| P0 / Dynamic sample | `terrain_length` | `8 m` | `8 m` |
| P0 / Dynamic sample | Remaining-distance katsayısı | `.625` | `.625` |
| P0 / Dynamic sample | Minimum velocity bound | `remaining_dist / (remaining_steps × dt)` | Aynı |
| P0 / Dynamic sample | X edge case | Reset’in son adımında `remaining_steps≈0` durumunda bölme riski mevcut | `remaining_steps <= 0` için lower bound explicit `0` yapılıyor |
| P0 / Command mix | Small XY command threshold | XY norm `≤ .2` ise uniform resampling branch’inde sıfırlanıyor | Aynı |
| P0 / Limit mix | Limited velocity olasılığı | `0.2` | `0.2` |
| P0 / Limit mix | Zero command olasılığı | Iteration curriculum ile `0 → .1` | Aynı |
| P0 / Limit mix | Zero command curriculum | `0–1500` iteration, `0.0 → 0.1` | Aynı |
| P0 / Limit mix | Zero command’da yaw-limit olasılığı | `0.2` | Aynı |
| P0 / Limit mix | Limit combinations | X `[-1,1]`, Y `[-1,1]`, yaw `[-1,0,1]` | Aynı |
| P0 / Limit mix | Limit invert | Açık | Açık |
| P0 / Limit mix | Limit sonrası heading | `stop_heading_at_limit=True` | Aynı |
| P1 / Accumulation | XY command accumulation | Her resampling’de current XY command ekleniyor | Aynı |
| P1 / Accumulation | Accumulation reset | Env reset’inde sıfırlanıyor | Aynı |
| P1 / Resampling order | Reward vs command | Önce reward, sonra command resample | Aynı |
| P1 / Reset order | Dynamic bound hesabı | Kaynakta reset sırası nedeniyle max-episode edge case’i var | Y reset öncesi `episode_length_buf=0` yaparak kaynak sırasını koruyor |
| P1 / Terrain owner | Terrain-family limits | Base environment command range’leri kullanıyor | WTY mixin semantic terrain ID’ye göre clamp/intersect ediyor |
| P1 / Terrain owner | Global/terrain range birleşimi | Global range ve terrain max range birlikte kullanılıyor | Aynı |
| P2 / Legacy curriculum | Host performance-based curriculum | Kaynakta WTY iteration listesi esas | Y `commands.curriculum=False`; host legacy performance path kapalı |
| P2 / Legacy standstill | Host `zero_cmd_prob` | Kaynak config’te bu host alanı kullanılmıyor | `zero_cmd_prob=0`; yalnız vendored zero curriculum kullanılıyor |
| P2 / Turn-over commands | `turn_over_zero_time` | Backflip `5 s`, sideflip `3 s` tanımlı | Alan etkili değil; Y turn-over machinery taşımıyor |
| P2 / Heading | Stop-heading field | Tanımlı ama `heading_command=False` olduğu için effective değil | Aynı |

Command sistemi sayısal olarak Y’de iyi taşınmış görünüyor. Y’nin eklediği önemli davranış, dynamic resampling reset edge case’inde `remaining_steps <= 0` durumunu güvenli şekilde ele alması.

---

## 6. Domain randomization ve control randomization

| Öncelik / grup | Parametre | X — `go2_rl_gym` | Y — LeggedGym-Ex |
|---|---|---|---|
| P0 / Friction | Friction randomization | Açık | Açık |
| P0 / Friction | X raw friction range | `[0,2]` | — |
| P0 / Friction | X ground friction | `1.0` | — |
| P0 / Friction | X effective fiziksel yorum | PhysX combine ile yaklaşık effective `[.5,1.5]` | — |
| P0 / Friction | Y raw link friction range | — | `[.5,1.5]` absolute link friction |
| P0 / Friction | Y terrain friction | — | `0.5` |
| P0 / Friction | Y Genesis combine | — | `max(link, ground, 1e-2)` |
| P0 / Friction | Effective support | Yaklaşık `[.5,1.5]` | `[.5,1.5]` |
| P0 / Friction | Randomization timing | Env/shape creation sırasında; 64 bucket | Genesis startup sırasında; env başına draw |
| P0 / Friction | Shape/link broadcast | Her env’de shape’lere aynı friction | Her env’de link’lere aynı friction ratio |
| P0 / Mass | Base mass randomization | Açık, added mass `[-1,1] kg` | Açık, `[-1,1] kg` |
| P0 / Mass | Base mass timing | Asset/creation path | Genesis startup |
| P0 / Mass | Link mass randomization | Açık, multiplier `[.9,1.1]` | Uygulanmıyor; Y host Genesis path’inde equivalent alan yok |
| P0 / COM | Base COM randomization | Açık, her eksende `[-.03,.03] m` | Açık, X/Y/Z `[-.03,.03] m` |
| P0 / COM | COM timing | Asset creation path | Genesis startup |
| P0 / PD | PD gain randomization | Açık | Açık |
| P0 / PD | Kp multiplier | `[.9,1.1]` | `[.9,1.1]` |
| P0 / PD | Kd multiplier | `[.9,1.1]` | `[.9,1.1]` |
| P0 / PD | Gain granularity | Per-DOF draw | `pd_gain_scalar=False`, per-DOF draw |
| P0 / PD | Gain timing | Reset sırasında | Startup ve reset path’inde |
| P0 / Action delay | Random action delay | Açık | Açık |
| P0 / Action delay | Delay unit | Simulator substep | Simulator substep |
| P0 / Action delay | Range | `{0,1,2,3,4}` substep | `{0,1,2,3,4}` substep |
| P0 / Action delay | Delay zamanı | Her control step’te yeniden draw | Her control step’te yeniden draw |
| P0 / Action delay | Fiziksel etki | İlk `k` substep previous action, kalan current action | Aynı |
| P0 / Action delay | Maksimum latency | `4 × .005 = 20 ms` | `20 ms` |
| P0 / Action delay | Legacy queue range | — | `[0,1]` alanı var fakat substep mode’da kullanılmıyor |
| P0 / Motor | Motor zero offset | Açık, `[-.035,.035] rad` | Uygulanmıyor |
| P0 / Motor | Motor strength | Açık, `[.8,1.2]` torque multiplier | Uygulanmıyor |
| P0 / Restitution | Robot restitution | Açık, `[0,.5]` | Config’te effective olarak kapalı |
| P0 / Restitution | Genesis restitution | — | `_randomize_restitution` no-op; fiziksel olarak uygulanmıyor |
| P1 / Push | Base push | Açık | Açık |
| P1 / Push | Push interval | `4 s` | `4 s` |
| P1 / Push | Push period in control steps | `200` | `200` |
| P1 / Push | Linear XY push range | `[-.4,.4] m/s` | Aynı |
| P1 / Push | Angular XYZ push range | `[-.6,.6] rad/s` | Aynı |
| P1 / Push | Push semantics | Velocity overwrite; selected env’lere apply | Velocity overwrite; selected env’lere apply |
| P1 / Push | Linear Z | Değişmiyor | Değişmiyor |
| P1 / Push | Joint velocities | Değişmiyor | Değişmiyor |
| P1 / Push | Push frame | World-frame root velocity | World-frame base velocity |
| P1 / Push | Trigger ownership | Her env’in kendi episode step’i | Her env’in kendi episode step’i |
| P1 / Link DR | Random link push | Kaynak GO2 task’ında aktif değil | `push_links=False` |
| P2 / Joint DR | Joint armature randomization | Yok | `False` |
| P2 / Joint DR | Joint friction randomization | Yok | `False` |
| P2 / Joint DR | Joint damping randomization | Yok | `False` |
| P2 / Camera DR | Camera position | Yok | `False` |
| P2 / Camera DR | Camera orientation | Yok | `False` |
| P2 / Asset | Nominal joint armature | Generic source `0` | `[.01] × 12` |
| P2 / Asset | DOF velocity limits | Isaac Gym asset/URDF’den | Explicit `[30.1,30.1,15.7] × 4` |
| P2 / DR state | DR state persistence | Bazı değerler creation-time, motor değerleri reset-time | Friction/mass/COM çoğunlukla startup; PD reset ile tekrar draw ediliyor |
| P2 / DR | Randomization reproducibility | Simulator/NumPy/PyTorch RNG’ye bağlı | Genesis/PyTorch/Genesis RNG’ye bağlı; aynı seed ile aynı dağılım garanti değil |

### Friction farkının özeti

Sayısal destek aralıkları benzer görünse de uygulama semantiği farklı:

- X: PhysX’te robot shape friction `U[0,2]`, terrain friction `1.0`.
- Y: Genesis’te terrain friction `0.5`, link friction `U[0.5,1.5]`, combine `max(link, terrain)`.

Y’nin `[.5,1.5]` aralığı, X’in PhysX effective friction dağılımını yaklaşık korumak için seçilmiş; fakat bu bir fiziksel eşdeğerlik kanıtı değil.

---

## 7. Initial state, reset ve termination

| Öncelik / grup | Parametre | X — `go2_rl_gym` | Y — LeggedGym-Ex |
|---|---|---|---|
| P0 / Initial state | Base başlangıç pozisyonu | `[0,0,.42]` | `[0,0,.42]` |
| P0 / Initial state | Base başlangıç yaw | Her reset `U(-π,π)` | `yaw_random_scale=3.14`, yaklaşık `U(-3.14,3.14)` |
| P0 / Initial state | Roll randomization | `0` | `0` |
| P0 / Initial state | Pitch randomization | `0` | `0` |
| P0 / Initial state | Base linear velocity | Her eksen `U(-.5,.5)` | Aynı |
| P0 / Initial state | Base angular velocity | Her eksen `U(-.5,.5)` | Aynı |
| P0 / Initial state | Custom-origin XY jitter | `±1.0 m` | `±0.5 m` |
| P0 / Initial state | XY jitter farkı | Daha geniş başlangıç spawn alanı | X’in yarısı kadar alan |
| P0 / DOF reset | DOF position | `default_dof_pos × U(.5,1.5)` | Aynı; WTY mixin host additive reset’i override ediyor |
| P0 / DOF reset | DOF velocity | `0` | `0` |
| P0 / Reset | Command resampling | Reset sırasında | Reset sırasında |
| P0 / Reset | Episode length reset sırası | Kaynakta command resampling ile episode counter sırası edge case üretebilir | Y önce episode length’i sıfırlayıp kaynak ordering’i geri kuruyor |
| P0 / Termination | Termination body | `["base"]` | `["base"]` |
| P0 / Termination | Contact force threshold | `1.0 N` | `2.5 N` |
| P0 / Termination | Contact comparison | `force_norm > threshold` | `force_norm > threshold` |
| P0 / Termination | Termination zamanı | Aynı physics/control step içinde | Aynı |
| P0 / Termination | Tilt/projected-gravity termination | Kaynakta yorumlanmış/etkisiz | Host tilt path’i mixin tarafından bypass ediliyor |
| P0 / Termination | Consecutive fail counter | Yok | Host `fail_to_terminal_time_s` counter’ı bypass ediliyor |
| P0 / Termination | `fail_to_terminal_time_s` | Yok | Config’te `.1 s`, fakat MoE-CTS termination override’ında etkisiz |
| P0 / Timeout | Episode timeout | `episode_length_buf > max_episode_length` | Aynı |
| P0 / Timeout | Timeout duration | `25 s` | `25 s` |
| P0 / Termination buffer | `reset_buf` | Contact OR timeout | Contact OR timeout |
| P0 / Termination buffer | `fail_buf` | Contact/fail semantiği | `0/1` same-step base-contact flag |
| P1 / Reset | Terrain curriculum update | `cfg.terrain.curriculum=True` üzerinden | `cfg.terrain.curriculum=False` olduğu için mixin explicit çağırıyor |
| P1 / Terrain level | Promote | `max_move_distance > 4 m` | Aynı |
| P1 / Terrain level | Demote | Accumulated command kuralı | Aynı |
| P1 / Terrain level | Last-level recycle | Random lower level | Aynı |
| P1 / Push order | Push zamanı | Reward/reset sonrası, observation öncesi | Aynı |
| P1 / Observation order | Observation timing | Push sonrası obs | Push sonrası obs |
| P1 / DOF default pose | Front thigh | `.8` | `.8` |
| P1 / DOF default pose | Rear thigh | `1.0` | `1.0` |
| P1 / DOF default pose | Hips | FL/RL `+.1`, FR/RR `-.1` | Aynı |
| P1 / DOF default pose | Calves | `-1.5` | Aynı |
| P2 / Turn over | `turn_over` | `False`; kaynak machinery config’te mevcut | `False`; turn-over machinery port edilmemiş |
| P2 / Turn over | Backflip/sideflip initial proportions | `[0,.2,.8]` tanımlı fakat `turn_over=False` | Effective değil |
| P2 / Gravity guard | `max_projected_gravity` | Kaynak termination’da etkili değil | Host default `-.1`, fakat mixin contact-only termination ile etkisiz |

`2.5 N` termination threshold, X’e göre Y’de daha toleranslı bir base-contact termination oluşturuyor. Bu fark training reward curve, episode length, curriculum promotion/demotion ve student/teacher sample composition’ını doğrudan etkileyebilir.

---

## 8. Reward ve penalty parametreleri

Önce ortak önemli metadata:

| Öncelik / grup | Parametre | X — `go2_rl_gym` | Y — LeggedGym-Ex |
|---|---|---|---|
| P0 / Reward | Reward scale’lerin timestep ile çarpılması | Config scale’leri `dt=.02` ile çarpılıyor | Aynı |
| P0 / Reward | `only_positive_rewards` | `False` | `False` |
| P0 / Reward | Negative reward clipping | Yok | Yok |
| P0 / Reward | Termination reward | Termination scale effective olarak `0` | Effective olarak `0` |
| P0 / Reward | Reward curriculum update | PPO iteration başına | PPO iteration başına |
| P0 / Reward | Reward iteration hesabı | `common_step_counter // 24` | `common_step_counter // 24` |
| P0 / Tracking | Base tracking sigma | `.25` | `.25` |
| P0 / Tracking | Dynamic sigma | Açık | Açık |
| P0 / Tracking | Terrain-level sigma scaling | `exp((level+1)/10)-1`, max `1` | Aynı |
| P0 / Tracking | Linear velocity sigma range | `.5 → 1.5 m/s` | Aynı |
| P0 / Tracking | Angular velocity sigma range | `1.0 → 2.0 rad/s` | Aynı |
| P0 / Reward metadata | `max_contact_force` | `147 N` | `147 N` |
| P0 / Reward metadata | `min_legs_distance` | `.1 m` | `.1 m` |
| P1 / Reward metadata | `max_contact_force` gerçekten aktif mi? | Hayır; `feet_contact_forces` scale’i yok | Hayır; aynı |
| P1 / Reward metadata | `min_legs_distance` gerçekten aktif mi? | Hayır; `legs_distance` scale’i yok | Hayır; aynı |

### Aktif reward scale’leri

| Öncelik / grup | Reward / penalty | X — `go2_rl_gym` | Y — LeggedGym-Ex |
|---|---|---|---|
| P0 / Tracking | `tracking_lin_vel` | `+1.0`; `exp(-ex²/σx - ey²/σy)` | Aynı dynamic sigma ile |
| P0 / Tracking | `tracking_ang_vel` | `+0.5`; `exp(-eyaw²/σ)` | Aynı |
| P0 / Penalty | `lin_vel_z` | `-2.0`; `base_lin_vel_z²` | Aynı |
| P0 / Penalty | `lin_vel_z` curriculum | Iteration `0→1500`, multiplier `1→0` | Aynı |
| P0 / Penalty | `ang_vel_xy` | `-0.05`; XY angular velocity square sum | Aynı |
| P0 / Penalty | `dof_acc` | `-2.5e-7`; control-step DOF acceleration square sum | Aynı effective window |
| P0 / Penalty | `dof_power` | `-2e-5`; `sum(abs(torque × dof_vel))` | Aynı |
| P0 / Penalty | `torques` | `-1e-4`; torque square sum | Aynı |
| P0 / Penalty | `correct_base_height` | `-1.0`; base-height target square error | Aynı |
| P0 / Penalty | Base-height curriculum | Iteration `0→5000`, multiplier `1→10` | Aynı |
| P0 / Penalty | `action_rate` | `-0.01`; `(last_action-action)²` | Aynı |
| P0 / Penalty | `action_smoothness` | `-0.01`; `(a_t - 2a_{t-1}+a_{t-2})²` | Aynı |
| P0 / Penalty | `collision` | `-1.0`; penalized thigh/calf links force norm `> .1 N` sayımı | `-1.0`; config threshold `.1 N` üzerinden aynı sayım |
| P0 / Penalty | `dof_pos_limits` | `-2.0` | Aynı |
| P0 / Penalty | Soft DOF position limit | `.9` | `.9` |
| P0 / Penalty | `feet_regulation` | `-0.05`; foot XY velocity ve foot-height exponential regularizer | Aynı |
| P0 / Penalty | `hip_to_default` | `-0.05`; dört hip’in absolute position error toplamı | Aynı |
| P1 / Acceleration | X dacc reference window | `last_dof_vel` bir önceki control step’e ait; `.02 s` | `_wty_last_dof_vel` ile bir önceki control step; `.02 s` |
| P1 / Acceleration | Host default dacc farkı | — | Y host default’u substep `.005 s` penceresi kullanacağı için Y bunu override ediyor |
| P1 / Collision | X collision threshold | `0.1 N` | `0.1 N` |
| P1 / Collision | Host default collision threshold | — | Host base normalde `10 N`; WTY override bunu kullanmıyor |
| P1 / Base height | Ground estimate | Masked height scan ortalaması | Aynı masked height scan |
| P1 / Feet regulation | Base height target | `.38 m` | `.38 m` |
| P1 / Feet regulation | Foot velocity frame | World-frame foot XY velocity | Genesis world-frame foot XY velocity |
| P1 / Hip | Hip indices | `[0,3,6,9]` | `[0,3,6,9]` |
| P2 / Inactive | `orientation` | Scale yok/effective değil | Scale yok/effective değil |
| P2 / Inactive | `dof_vel` | Scale yok/effective değil | Scale yok/effective değil |
| P2 / Inactive | `base_height` legacy reward | Scale yok; `correct_base_height` kullanılıyor | Scale yok; `correct_base_height` kullanılıyor |
| P2 / Inactive | `feet_air_time` | Scale yok | Scale yok |
| P2 / Inactive | `stumble` | Scale yok | Scale yok |
| P2 / Inactive | `standstill` | Scale yok | Scale yok |
| P2 / Inactive | `feet_contact_forces` | Scale yok | Scale yok |
| P2 / Inactive | `legs_distance` | Scale yok | Scale yok |
| P2 / Turn-over reward | `upright` | `turn_over_scales.upright=1`, fakat `turn_over=False` | Turn-over reward machinery yok; effective değil |

Reward katsayıları ve aktif reward formülleri Y’de oldukça yakın. En önemli reward-level farkları:

- Y’nin termination eşiği farklı.
- Y Genesis contact measurement’ı üzerine collision threshold `.1 N` koyuyor.
- Y `dof_acc` için host’un yanlış substep penceresini kullanmıyor; control-rate tracker ile X’e yaklaştırıyor.
- Simulator dynamics farkı nedeniyle aynı reward formülü aynı numeric reward distribution’ını üretmeyebilir.

---

## 9. Observation normalization ve noise

| Öncelik / grup | Parametre | X — `go2_rl_gym` | Y — LeggedGym-Ex |
|---|---|---|---|
| P0 / Normalization | Linear velocity scale | `2.0` | `2.0` |
| P0 / Normalization | Angular velocity scale | `.25` | `.25` |
| P0 / Normalization | DOF position scale | `1.0` | `1.0` |
| P0 / Normalization | DOF velocity scale | `.05` | `.05` |
| P0 / Normalization | Height measurement scale | `2.5` | `2.5` |
| P0 / Normalization | Action scale in obs | `1.0` | `1.0` |
| P0 / Normalization | Observation clip | `100` | `100` |
| P0 / Normalization | Action clip | `100` | `100` |
| P0 / Noise | Add noise | Açık | Açık |
| P0 / Noise | Noise level | `1.0` | `1.0` |
| P0 / Noise | Angular velocity noise | `.2 × .25 = .05` effective | Aynı |
| P0 / Noise | Gravity noise | `.05` | `.05` |
| P0 / Noise | Command noise | `0` | `0` |
| P0 / Noise | DOF position noise | `.01` | `.01` |
| P0 / Noise | DOF velocity noise | `1.5 × .05 = .075` effective | Aynı |
| P0 / Noise | Action noise | `0` | `0` |
| P0 / Noise | Privileged observation noise | Yok; privileged clean kalıyor | Yok; privileged clean kalıyor |
| P1 / Noise | Noise distribution | Uniform `[-1,1] × noise_scale` | Aynı |
| P1 / Noise | Noise application timing | Privileged oluşturulduktan sonra actor obs’e | Aynı |
| P1 / Height | Height clipping | `[-1,1]` | `[-1,1]` |

Bu bölümde X/Y arasında anlamlı numeric fark görmedim.

---

## 10. Asset, robot ve düşük seviyeli dünya parametreleri

| Öncelik / grup | Parametre | X — `go2_rl_gym` | Y — LeggedGym-Ex |
|---|---|---|---|
| P0 / Asset | Robot | Unitree Go2 | Unitree Go2 |
| P0 / Asset | URDF path | `resources/robots/go2/urdf/go2.urdf` | `resources/robots/unitree_robotics/go2/urdf/go2.urdf` |
| P0 / Asset | Genesis XML | Yok | `resources/robots/unitree_robotics/go2/go2.xml` |
| P0 / Asset | Genesis asset loading | — | XML/MJCF path zorunlu |
| P0 / Asset | Foot name | `foot` | `foot` |
| P0 / Asset | Penalized contacts | `["thigh","calf"]` | `["thigh","calf"]` |
| P0 / Asset | Termination contacts | `["base"]` | `["base"]` |
| P0 / Asset | Self-collision flag | `0`, self collision açık | `0`, Genesis self collision açık |
| P1 / Asset | Base link name | Asset/URDF çözümlemesine bağlı | Explicit `"base"` |
| P1 / Asset | DOF names | Config’te açık liste yok; URDF sırasına bağlı | Explicit 12-DOF listesi |
| P1 / Asset | DOF order | URDF ordering belirleyici | `FR`, `FL`, `RR`, `RL`; hip/thigh/calf |
| P1 / Asset | DOF armature | Generic source default `0` | `.01 × 12` |
| P1 / Asset | DOF velocity limits | URDF/Isaac Gym asset’ten | Explicit `[30.1,30.1,15.7] × 4` |
| P1 / Control | Control type | `P` | `P` |
| P1 / Control | Stiffness | `20.0` | `20.0` |
| P1 / Control | Damping | `.5` | `.5` |
| P1 / Control | Action scale | `.25` | `.25` |
| P1 / Control | Target joint angle | `default + .25 × action` | Aynı |
| P2 / Asset | Fixed joint collapse | Isaac Gym generic default | Genesis `links_to_keep` ile asset’e bağlı |
| P2 / Asset | Links kept | Isaac Gym default path | Foot link’leri explicit kept |
| P2 / Asset | Robot density/damping | Isaac Gym asset defaults | XML/MJCF değerlerine bağlı |
| P2 / Asset | Max velocity limits | PhysX asset defaults | Genesis XML/solver davranışına bağlı |

Burada config’teki robot eklem isimleri ve default pozisyonlar eşleşiyor; ancak X’in implicit URDF DOF order’ı ile Y’nin explicit `dof_names` order’ı mutlaka runtime’da ayrıca doğrulanmalı. Yalnızca default joint dictionary’sinin sırası buna kanıt değil.

---

## 11. Checkpoint, inference ve export farkları

| Öncelik / grup | Parametre | X — `go2_rl_gym` | Y — LeggedGym-Ex |
|---|---|---|---|
| P0 / Checkpoint | Model state key | `model_state_dict` | `model_state_dict` |
| P0 / Checkpoint | RL optimizer key | `optimizer1_state_dict` | `optimizer_state_dict` |
| P0 / Checkpoint | Student optimizer key | `optimizer2_state_dict` | `history_encoder_optimizer_state_dict` |
| P0 / Checkpoint | Model encoder key names | `teacher_encoder`, `student_moe_encoder` | `privilege_encoder`, `history_encoder` |
| P0 / Resume | Aux optimizer resume | İki optimizer explicit yükleniyor | Auxiliary optimizer `_aux_optimizers()` üzerinden yükleniyor |
| P0 / Deploy | Inference policy | `model.act_inference` | `actor_critic.act_student` |
| P0 / Deploy | Inference history | Model internal persistent history buffer | `HistoryObsAdapter` history sağlıyor |
| P0 / Deploy | Deploy state prefixes | Kaynak actor + student model isimleri | `("actor.", "history_encoder.")` |
| P0 / Import | X checkpoint’i Y’ye doğrudan yükleme | Kaynak isimleriyle | State-key remap gerektiriyor |
| P1 / Checkpoint | Iteration field | `iter` | `iter` |
| P1 / Checkpoint | Iteration semantics | Kaynakta yalnız integer iteration | Y `completed_updates_v2` metadata taşıyor |
| P1 / Checkpoint | Training seed metadata | Ayrı metadata yok | `training_seed` saklanıyor |
| P1 / Checkpoint | Schedule metadata | Ayrı metadata yok | Active command schedule stage/range saklanıyor |
| P1 / Checkpoint | Best eval metadata | Yok veya infos’a bağlı | `best_eval_score`, `best_tracking_key` |
| P1 / Checkpoint | SPNTE metadata | Yok | `best_spnte_score`, `best_spnte_key` |
| P2 / Logging | MoE expert usage | Temel update loss output’u | TensorBoard/terminal expert usage ve entropy |
| P2 / Evaluation | Student-only eval | Model inference history’si | Adapter üzerinden student-only eval |
| P2 / Evaluation | Privileged obs deployment | Training-only | Training-only; deploy prefix’lerinden çıkarılmış |

Y checkpoint formatı kaynak checkpoint formatından daha zengin, fakat bunun bedeli upstream checkpoint’lerin doğrudan `load_state_dict` ile yüklenememesi. Özellikle encoder adları ve optimizer key’leri farklı.

---

## En önemli farkların kısa önem sırası

| Sıra | Fark | Muhtemel etkisi |
|---:|---|---|
| 1 | Isaac Gym/PhysX/trimesh → Genesis/Newton/heightfield | Tüm dinamikler, contact force, friction, terrain interaction ve öğrenme hızı değişebilir |
| 2 | Termination threshold `1.0 N → 2.5 N` | Base contact sonrası episode uzunluğu, reward, curriculum ve data distribution değişir |
| 3 | ~~X bağımsız critic permutation, Y aligned critic minibatch~~ — **iptal**, yanlış atıf; iki taraf da role-aligned | Yok; value loss sample contract aynı |
| 4 | X `±1 m` spawn jitter, Y `±.5 m` | Initial-state distribution ve terrain-origin exploration değişir |
| 5 | X seed `0`, Y seed `1` | Exact reproducibility ve erken training trajectory değişir |
| 6 | Link mass/motor zero/motor strength/restitution DR farkları | X’in policy’si daha geniş veya farklı fizik dağılımı görür |
| 7 | Friction raw/config semantics farklı | Aynı görünen support aralığı farklı combine kurallarıyla uygulanıyor |
| 8 | X implicit URDF DOF order, Y explicit Genesis DOF order | Yanlış order varsa observation/action ve default pose tamamen bozulabilir |
| 9 | Checkpoint/state-key/optimizer ABI farkı | Resume/import/export doğrudan uyumlu değil |
| 10 | Y control-rate DOF acceleration düzeltmesi | Y, X’in 20 ms acceleration penalty’sine daha yakın |
| 11 | Y dynamic-resample reset edge-case düzeltmesi | X’teki son-step division problemine karşı daha güvenli davranış |
| 12 | Terrain geometry formülleri | Büyük ölçüde aynı |
| 13 | Command limits/curriculum | Büyük ölçüde aynı |
| 14 | Active reward scales/formülleri | Büyük ölçüde aynı; simulator-derived reward values yine farklı olabilir |

## Son hüküm

Y implementasyonu; observation ABI, network topology, terrain family bankı, command sistemi, reward scale’leri ve temel MoE loss açısından X’i ciddi ölçüde takip ediyor. Ancak aşağıdaki üç alan nedeniyle “X’in birebir implementasyonu” olarak etiketlenmemeli:

1. PPO/storage sözleşmesi birebir değil — ancak yalnızca yapısal düzeyde (env sırası + role index’leri, tuple paketlemesi, encoder/optimizer isimleri); sample selection ve loss aggregation semantiği eşleşiyor.
2. Simulator/world sözleşmesi birebir değil.
3. Bazı DR, termination ve reset parametreleri bilinçli olarak değiştirilmiş veya uygulanmamış.

En kritik parity kararı şu olurdu:

- Eğer hedef “X’in method mimarisini Genesis’e taşımak” ise Y’nin network/observation/method tarafı oldukça yakın.
- Eğer hedef “X ile aynı eğitim davranışını üretmek” ise termination threshold, root spawn jitter, friction ve eksik DR alanları için açık bir parity sözleşmesi oluşturmak gerekir. Advantage normalization ve critic minibatch alignment bu listenin dışında kalır; MoE-CTS’te iki taraf da global normalize eder ve iki taraf da role-aligned critic sample kullanır.
- Eğer hedef “X’in source-faithful portu” ise PPO update semantics tarafında yeniden değerlendirilecek bir kalem kalmıyor; geriye kalan farklar simulator/world sözleşmesi ile DR/termination/reset parametreleridir.
