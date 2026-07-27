# MLP–Oracle headroom report input

`python -m legged_gym.scripts.eval.headroom_report --input normalized.json --output report.html`

Bu araç kampanya ham çıktılarını, checkpoint seçimlerini veya güvenlik kapılarını
yeniden yorumlamaz. Girdi, bunlardan sonra oluşturulmuş normalize edilmiş dünya
tablosudur. Tracking headroom yalnızca şu açık formülle hesaplanır:

`(MLP tracking error - method tracking error) / (MLP tracking error - Oracle tracking error)`

Tracking error düşük olması daha iyi bir metriktir. `fall_rate` ve
`achieved_speed_ratio` ayrı survival tablosuna gider; hiçbir zaman bu orana
birleştirilmez.

## JSON

```json
{
  "experiment": {
    "name": "V4 sabit terrain karşılaştırması",
    "contract": "Aynı terrain, command bank, eval seed ve eğitim bütçesi.",
    "seed_count": 3,
    "tracking_metric": "Mean planar tracking error (m/s)",
    "limitations": ["Oracle seçimi ayrı bir güvenlik kapısından geçmiştir."]
  },
  "worlds": [
    {
      "world": "Stairs / L7 / 1.0 m/s",
      "id_ood": "OOD",
      "include": true,
      "exclusion_reason": "",
      "tracking_error": {"MLP": 0.82, "Oracle": 0.50, "DreamWaQ": 0.61, "HIM": 0.58},
      "fall_rate": {"MLP": 0.04, "Oracle": 0.01, "DreamWaQ": 0.02, "HIM": 0.01},
      "achieved_speed_ratio": {"MLP": 0.88, "Oracle": 0.97, "DreamWaQ": 0.94, "HIM": 0.96},
      "seed_consistency": {
        "DreamWaQ": {"better_seeds": 3, "total_seeds": 3},
        "HIM": {"better_seeds": 2, "total_seeds": 3}
      }
    }
  ]
}
```

`include: true` tek başına yeterli değildir: MLP tracking error Oracle'dan
büyük olmalıdır. Aksi durumda dünya dışlananlar tablosuna girer. `include:
false` ise `exclusion_reason` verilmelidir.

## CSV

Bir satır bir dünyadır. Asgari başlıklar şunlardır:

```text
world,id_ood,include,exclusion_reason,tracking_error_mlp,tracking_error_oracle,tracking_error_dreamwaq,tracking_error_him,fall_rate_mlp,fall_rate_oracle,fall_rate_dreamwaq,fall_rate_him,achieved_speed_ratio_mlp,achieved_speed_ratio_oracle,achieved_speed_ratio_dreamwaq,achieved_speed_ratio_him,seed_consistency_dreamwaq,seed_consistency_him
```

CSV metadatası komut satırında verilir:

```bash
python -m legged_gym.scripts.eval.headroom_report \
  --input worlds.csv --output headroom.html \
  --name "V4 sabit terrain karşılaştırması" \
  --contract "Aynı terrain, command bank, eval seed ve eğitim bütçesi." \
  --seed-count 3 --tracking-metric "Mean planar tracking error (m/s)"
```

`seed_consistency_*` alanı `3/3` biçiminde ya da boş bırakılabilir. `id_ood`
yalnızca `ID` veya `OOD` değerini alır.
