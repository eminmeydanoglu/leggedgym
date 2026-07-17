# V4: orta terrain + terrain-map oracle kampanyası

V4, bitmiş V3 kampanyasını değiştirmez. Aynı altı yöntemi, V3'ün P5 ve tek
episode-içi mass/CoM switch kontratıyla; fakat her ortamda **sabit orta seviye**
terrain üzerinde yeniden eğitir.

## Sabit eğitim kontratı

| Bileşen | V4 kararı |
|---|---|
| Terrain üretimi | 10 terrain sütunu, 10 zorluk satırı; sütunlarda terrain türü karışımı |
| Kullanılan satır | Her ortam için sabit satır `5`; generator tanımında difficulty `5 / 10 = 0.5` |
| Terrain ilerleme curriculum | Kapalı; generator grid'i kurar ama öğrenen performansına göre satır değişmez |
| Terrain çeşitleri | Eğimi, random-uniform, merdivenleri ve discrete-obstacle zeminleri |
| Fizik | V3 ile aynı: `P5=[friction,mass,com_x,com_y,com_z]`, `[-2,+5] kg`, `±0.08 m`, episode başına bir switch |
| Sıradan yöntemler | V3 aktör/critic girişleri aynen korunur; terrain map aktöre verilmez |
| RMA teacher | Gerçek `P5 + velocity + 17×11` height map görür; student yine yalnız 20×45 proprio history'den 8D latent üretir |
| Oracle | 20-frame proprio + gerçek velocity + gerçek P5 + 17×11, noise-free, yaw-aligned yerel height map |

Bu tasarımda oracle'ın üstünlüğü iki parçaya ayrılır: gerçek fizik bilgisi ve
gerçek terrain algısı. Bu nedenle V4 sonucu, *history-only online adaptation
terrain kaynaklı kaybın ne kadarını kapatıyor; tam terrain bilgisi ile kalan
üst sınır nedir?* sorusunu cevaplar. Oracle, deploy edilebilir bir yöntem değil,
terrain-aware üst sınırdır.

## Görevler

| Rol | V4 görevi | Aktör girişi |
|---|---|---|
| DR-MLP | `go2_v4_mlp` | 45D proprio |
| Explicit SysID | `go2_v4_sysid` | 20×45 history |
| RMA | `go2_v4_rma` | Student: 45D proprio + 20×45 history; teacher: P5 + velocity + 187D height map |
| DreamWaQ | `go2_v4_dreamwaq` | 45D proprio + 5×45 history |
| HIM | `go2_v4_him_fixed` | 6×45 history |
| Terrain-aware oracle | `go2_v4_superset_oracle` | 20×45 + 3 velocity + P5 + 187 height sample = 1095D |

## Çalıştırma sırası

Önce lokal kontrat ve Genesis smoke kontrolü:

```bash
cd /home/emin/code/online-estimation/genesis-wp/LeggedGym-Ex
env SIMULATOR=genesis .venv/bin/python -m unittest tests/test_v3_training_contract.py tests/test_v4_training_contract.py
env SIMULATOR=genesis .venv/bin/python tests/v4_runtime_smoke.py
```

Altay'da her görev için V3 ile aynı seed/bütçe kullanılır; yalnız task adı V4,
deney kökü otomatik olarak `logs/go2_v4_medium_terrain/` olur:

```bash
env SIMULATOR=genesis WANDB_MODE=disabled .venv/bin/python -u legged_gym/scripts/train.py \
  --task go2_v4_rma --headless --seed 1 --num_envs 4096 --max_iterations 3000
```

İlk dört GPU'luk dalga için önerilen eşleştirme: `mlp/s1`, `sysid/s1`,
`rma/s1`, `dreamwaq/s1`; sonra `him/oracle/s1` ve seed 2'ler. V3 evaluation
YAML'sinin model satırları V4 task/run kökleriyle kopyalanarak aynı statik ve
mid-episode switch scorecard uygulanmalıdır. V4 terrain'i görev konfigürasyonun
parçası olduğu için, eval sırasında ayrıca terrain override verilmez.

## Kabul kapısı

1. Her iki kontrat testi geçer; V3 hâlâ flat/map-free kalır.
2. V4 runtime smoke her görevde terrain level=`5`, live mass/CoM switch ve
   beklenen aktör/critic boyutlarını doğrular.
3. İlk seed'de oracle'ın eğitimi stabil değilse, bütün kampanyayı başlatmadan
   önce yalnız terrain şiddeti/rewardlar incelenir; yöntemlere ayrı terrain
   curriculum uygulanmaz.
4. İki seed tamamlandıktan sonra V3 scorecard hücrelerinin aynısı V4 için
   üretilir ve hem mutlak skor hem `MLP→oracle` normalize headroom raporlanır.
