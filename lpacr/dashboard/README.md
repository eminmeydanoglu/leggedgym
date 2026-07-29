# Curriculum Atlas

LP-ACRL ve diğer curriculum yöntemleri için canlı, kalıcı ve training kodundan
bağımsız görev-dağılımı dashboard'u.

## Çalıştırma

Terminal 1:

```bash
cd /home/emin/code/online-estimation/genesis-wp/LeggedGym-Ex/lpacr/dashboard
npm start
```

Arayüz: <http://127.0.0.1:8765>

Gerçek training kodu hazır değilken örnek veri akışı:

```bash
cd /home/emin/code/online-estimation/genesis-wp/LeggedGym-Ex/lpacr/dashboard
npm run demo
```

Her frame `data/<run_id>/frames.ndjson` içine append-only olarak yazılır.
Sunucu veya browser kapatılsa bile geçmiş korunur. Dashboard açıldığında kayıtlı
run'ları bulur; `History` modu ve alt slider ile bütün geçmiş oynatılabilir.

Farklı bir veri dizini veya port kullanılabilir:

```bash
LPACRL_DATA_DIR=/path/to/runs LPACRL_PORT=9000 npm start
```

## Training entegrasyonu

`plugger.py` yalnızca Python standart kütüphanesini kullanır ve ağ I/O'sunu arka
plan thread'inde yapar. Training loop'u dashboard'u beklemez.

```python
from lpacr.dashboard.plugger import CurriculumDashboardPlugger, TaskSpace

task_space = TaskSpace(
    dimensions=("terrain_type", "terrain_level", "vx_bin", "yaw_bin"),
    coordinates={
        "terrain_type": ["descending", "ascending", "rough", "uphill", "downhill"],
        "terrain_level": ["L1", "L2", "L3", "L4"],
        "vx_bin": ["0–0.5", "0.5–1.0", "1.0–1.5", "1.5–2.0", "2.0–2.5"],
        "yaw_bin": ["0–0.5", "0.5–1.0", "1.0–1.5", "1.5–2.0", "2.0–2.5", "2.5–3.0"],
    },
)

dashboard = CurriculumDashboardPlugger("lpacrl-seed-1", task_space)

# Her curriculum update'inden sonra.
# `step` is the publisher's timeline key (V5 UED uses global_control_steps,
# not PPO iteration). The UI labels it "control step".
dashboard.log(
    step,  # e.g. global_control_steps
    {
        "performance": reward_tensor,
        "learning_progress": lp_tensor,
        "sampling_probability": probability_tensor,
        "success_rate": success_tensor,
        "sample_count": sample_count_tensor,
    },
)
```

Tensorlar `task_space.dimensions` sırasına göre C-order düzleştirilir. NumPy,
PyTorch CPU/GPU tensorları ve Python listeleri kabul edilir. PyTorch tensorları
plugger içinde `detach().cpu()` ile ayrılır.

Training sonunda bekleyen son frame'leri göndermek için:

```python
dashboard.close()
```

### V5 UED gerçek eğitim entegrasyonu

`go2_v5_uniform`, `go2_v5_lpacrl` ve `go2_v5_alp` için köprü artık gerçek
`EpisodeCurriculum.advance()` sonucunu yayınlar; demo verisi üretmez.
**Varsayılan olarak açıktır** (2026-07-27'den itibaren): `train.py` her UED
arm'ı için otomatik olarak `create_v5_dashboard_bridge` çağırır ve
`http://127.0.0.1:8765`'e publish etmeye çalışır. Sunucu ayakta değilse bu
tamamen zararsızdır -- publish thread'i arka planda sessizce (saniyede bir)
retry eder, eğitimi asla bloklamaz veya çökertmez (bkz.
`lpacr/dashboard/plugger.py` `_worker`). `go2_v5_handcrafted` gibi UED
olmayan arm'larda `create_v5_dashboard_bridge` zaten no-op döner.

Dashboard'u izlemek için önce sunucuyu açın, training'i olduğu gibi başlatın
(hiçbir ek bayrağa gerek yok):

```bash
cd /home/emin/code/online-estimation/genesis-wp/LeggedGym-Ex/lpacr/dashboard
npm start

cd /home/emin/code/online-estimation/genesis-wp/LeggedGym-Ex
env SIMULATOR=genesis WANDB_MODE=disabled .venv/bin/python -m legged_gym.scripts.train \
  --task go2_v5_lpacrl --headless
```

`--dashboard_url http://127.0.0.1:9000` ve `--dashboard_run_id benim-runim`
isteğe bağlı override'lardır (yoksa sırasıyla `LPACRL_DASHBOARD_URL` ve
otomatik `<task>-<run_dir>` kullanılır). Kapatmak için `--no_dashboard` CLI
bayrağı veya `LPACRL_DASHBOARD=0` ortam değişkeni kullanılır.

Her tamamlanmış stage için tek frame yazılır: `performance` (cell return),
`learning_progress`, `sampling_probability`, stage episode sayısı, assignment
ve completion sayıları. V5'in gözlenmemiş cell'leri `null` olarak kalır; sahte
sıfır return üretilmez. Run metadata, task/seed/algorithm ve V5 fingerprint'lerini;
frame metadata ise stage/revision, curriculum diagnostics ve ayrık standstill
bucket özetini taşır. NDJSON replay ve SSE frame'i bu opsiyonel metadata alanını
aynen korur.

`go2_v5_handcrafted` UED teacher/stage üretmediği için dashboard istenirse
anlamlı bir no-op yapar ve eğitim normal sürer.

### V6 frontier entegrasyonu

`go2_v6_*` arm'ları için `create_v5_dashboard_bridge` otomatik olarak
`FrontierDashboardBridge`'i seçer (curriculum'un task space'ine bakarak). Ek
bayrak gerekmez; v5 ile v6 aynı sunucuda yan yana yaşar ve üstteki run
seçicisinden geçiş yapılır.

**Çözünürlük.** Frame `(vx_bin, terrain_family, terrain_level)` = 4×6×10 = 240
hücredir. Terrain replikaları (aynı ailenin iki kolonu) curriculum kararı
*değil* — hücre seçildikten sonra uniform çekilirler — bu yüzden eksen
olmazlar; replika dengesi frame metadata'sındaki `replica_balance` alanında
raporlanır.

**Metrikler.** `FrontierCurriculum.cell_metrics()` hücre başına ~32 metrik
yayınlar: `state` (0 locked / 1 frontier / 2 mastered / 3 unstable),
`success_probability` ve stage başına `success_probability_delta`, pencere
doluluğu (`window_episode_count`, `episodes_until_eligible`), sampling kütlesi,
bucket provenance (`source_{frontier,replay,uniform}_{count,share}`), unlock ve
mastery stage damgaları, ve mastery kuralının attığı sürekli episode sinyalleri
(`mean_linear_error`, `mean_yaw_error`, `mean_episode_length`,
`mean_episodic_return`, `timeout_fraction`) — hepsi EWMA. Gözlenmemiş hücreler
uydurma sıfır değil, `null` kalır. Hepsi checkpoint'e yazılır, resume'da
korunur (`SCHEMA_VERSION = 2`).

**Görünüm.** Atlas varsayılan olarak X=level, Y=|vx|, panel=terrain family ile
açılır ve `state` metriğini gösterir: frontier iki eksende birden ilerlediği
için aile başına 10×4 ızgara "curriculum nerede" sorusunun tam cevabıdır.
Hangi metrik seçilirse seçilsin frontier hücreleri amber çerçeveyle
işaretlenir. Signal chain v6'da otomatik olarak
`success_probability → delta → sampling_probability` zincirine geçer (frontier
kuralının learning-progress sinyali yoktur). Alttaki panel v5'te ESS,
v6'da lifecycle sayılarını çizer.

Context manager zorunlu değildir; process normal kapanırsa `atexit` de son
frame'leri göndermeyi dener.

## Veri kontratı

Sunucuya doğrudan `POST /api/runs/<run_id>/frames` çağrısı da yapılabilir:

```json
{
  "step": 1250,
  "wall_time": 1784800000.0,
  "task_space": {
    "dimensions": ["terrain_type", "terrain_level", "vx_bin"],
    "coordinates": {
      "terrain_type": ["rough", "stairs"],
      "terrain_level": ["L1", "L2"],
      "vx_bin": ["0–0.5", "0.5–1.0"]
    }
  },
  "metrics": {
    "performance": [0, 1, 2, 3, 4, 5, 6, 7],
    "learning_progress": [0, 1, 0, -1, 2, 1, 0, 0],
    "sampling_probability": [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
  }
}
```

Her metrik dizisinin uzunluğu koordinat boyutlarının çarpımına eşit olmalıdır.
Yeni metrik adları dashboard'daki Metric seçicisine otomatik eklenir.

## View'lar

- Ana atlas: seçilebilir X/Y eksenleri, panel boyutu, kalan boyut için filtre ve
  seçilebilir metrik.
- Signal chain: performance, signed learning progress ve sampling probability.
- Distribution through time: her task boyutunun marjinal probability geçmişi.
- Hover inspection: hücrenin bütün koordinatları ve mevcut metrikleri.
- Live/history geçişi, frame slider'ı ve geçmiş oynatma.

Test:

```bash
npm test
```
