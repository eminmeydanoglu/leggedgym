# V5 / UED agent work and usage ledger

Bu dosya her implementer ve tester ajanının teslimatını ve fractal tarafından
raporlanan terminal kullanımını kaydeder. Fractal CLI token sayısını doğrudan
raporlamadığında token değeri tahmin edilmez; kesin terminal maliyeti yazılır.

| Ajan | Rol | Teslimat / doğrulama | Son commit | Test sonucu | Token | Terminal maliyet |
|---|---|---|---|---|---|---:|
| `main.v5_ued_01_spnte` | Implementer, Topic 01 | First-fall SPNTE ve V3/V4 eval wiring | `7aaa400` (`bcd3d36` implementasyon) | SPNTE 6/6, V3 eval 15/15 geçti; üç V2 resume fixture testi değişmemiş `self.device` hatasıyla kaldı | Fractal CLI raporlamadı | `$3.0214` |
| `main.v5_ued_02_teacher` | Implementer, Topic 02 | Saf NumPy TaskSpace, uniform/LP-ACRL/ALP teacher, checkpoint/RNG | `9a8bd70` (`1973c31` implementasyon) | Teacher kabul paketi 9/9 ve import guard geçti | Fractal CLI raporlamadı | `$2.1792` |
| `main.v5_ued_01_spnte_test` | Bağımsız tester, Topic 01 | V3 dinamik payload-switch artifact'ine eksik SPNTE accumulator/scale wiring'ini düzeltti | `de6e454` | SPNTE/V3/V4 focused 29 geçti; V2 non-resume 17 geçti; 3 V2 resume fixture hatası değişmemiş `main` archive'da aynı biçimde üretildi | Fractal CLI raporlamadı | `$1.2623` |
| `main.v5_ued_02_teacher_test` | Bağımsız tester, Topic 02 | TaskSpace identity/fingerprint, softmax taşması ve fail-closed checkpoint/batch doğrulamasını sertleştirdi | `e0b48f6` | Tester `test.sh`, doğrudan pytest ve AST import guard ile 12 test geçti | Fractal CLI raporlamadı | `$1.1506` |
| `main.v5_ued_orchestrator.v5_ued_03_genesis` | Implementer, Topic 03 | Genesis adapter, 84-task terrain grid, provenance reset state machine ve feature flag'ler | `2f9a009` | Node, odaklı Genesis smoke ve kabul testlerini geçtiğini raporladı; bağımsız tester iptalden önce çalışmadı | Fractal CLI raporlamadı | `$2.8729` |
| `main.v5_ued_orchestrator.v5_ued_04_validation` | Implementer, Topic 04 | Fixed 84x48 validation bank ve offline `best_spnte.pt` selector | `4cc397b` | Node, focused checkpoint-selection contract testlerinin geçtiğini raporladı; bağımsız tester iptalden önce çalışmadı | Fractal CLI raporlamadı | `$2.2128` |

## Entegrasyon durumu

- Topic 01 uygulaması ve tester düzeltmesi orchestrator dalına entegre edildi:
  `ee79e26`, `0bb5603`; entegrasyon sonrası `tests/test_spnte_metric.py` ve
  `tests/test_v3_eval.py` birlikte 22 test geçti.
- Topic 02 uygulaması ve tester düzeltmesi orchestrator dalına entegre edildi:
  `72c1b3d`, `833b571`; entegrasyon sonrası `tests/test_ued_teacher.py` 12 test
  geçti.
- Topic 03 ve Topic 04, completed child dallarından orchestrator dalına
  seed dosyaları taşınmadan squash-entegre edildi: `52f3900`, `711f9b4`.
- Birleşik test ortamı bu makinede yeniden kurulamadı: sistem Python'unda
  `pytest` yok; `uv`, Genesis ve IsaacGym'in uyumsuz sabit Torch extra
  bağımlılıkları nedeniyle çözüm bulamadı. Branch bazlı test kanıtları yukarıda
  korunur.
