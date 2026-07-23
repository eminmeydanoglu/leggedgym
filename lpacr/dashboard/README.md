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

# Her curriculum update'inden sonra:
dashboard.log(
    iteration,
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
