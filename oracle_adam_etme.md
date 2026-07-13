Bence şu sırayla ilerleyelim:

1. **Mevcut oracle’ı doğru adlandıralım: `P5 oracle` / `privileged-P policy`.**  
   Sebep: Bu politika tam simülatör durumunu değil, yalnızca `friction + mass + CoM` bilgisini görüyor. “Mutlak performans tavanı” demek fazla güçlü.

2. **Command curriculum’u bütün yöntemlerde eşitleyelim.**  
   Sebep: Wave‑1’de politikalar yüksek hız komutlarını farklı zamanlarda görmüş. Bu nedenle farkın `P` bilgisinden mi, eğitim dağılımından mı geldiği net değil.

3. **Checkpoint seçimini aynı tam command alanında yapalım.**  
   Sebep: `best.pt` dar hız alanında seçilirken ana değerlendirme `vx=1.0` üzerinden yapılmış. Eğitim, validation ve test alanları uyumlu olmalı.

4. **P5-MLP ve P5-oracle’ı aynı üç training seed ile tekrar eğitelim.**  
   Sebep: Tek seed, oracle avantajı hakkında güvenilir hüküm vermeye yetmez. Aynı seed’ler karşılaştırma gürültüsünü azaltır.

5. **Oracle headroom’unu gerçekten uyarıcı senaryolarda ölçelim.**  
   Sebep: Sürtünme, yavaş düz yürüyüşte neredeyse etkisiz kaldı. Yüksek hız, lateral/yaw komutu, ani command değişimi ve ek kütle daha ayrıştırıcı.

6. **Net bir karar kapısı kullanalım.**
   - Oracle tekrarlı biçimde MLP’den iyiyse → önce minimal `history → P5` estimator.
   - Oracle üstün değilse → P5 estimator/RMA/DreamWaQ eğitimine henüz geçmeyelim.

   Sebep: Estimator, en iyi durumda oracle’ın bilgisini yaklaşık üretir. Oracle’ın avantajı yoksa tahmin edilecek bilginin kanıtlanmış değeri de yoktur.

