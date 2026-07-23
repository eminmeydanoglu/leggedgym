## Fable'ın son önerdiği üst seviye plan

Gerçekleştirme planı istediğin için makalenin detaylarına (özellikle eval metodolojisine ve LP kestirimine) bir bakayım, sonra tam planı çıkarayım.Makalenin tam metnini okudum — eval protokolleri ve algoritmanın çekirdeği artık elimde. Üç sorunu sırayla, sonra planı.

## Şirketin muhtemelen ne yapıyor?

Kodları göremesen de sektör standardı reçete bellidir, ve makale de rakiplerini tek tek sayıyor. Neredeyse kesin şunlardan biri ya da karışımı: Rudin tarzı — paralel robotları önceden tanımlı terrain tiplerine dağıt, performans eşiğini geçen robotu aynı tip içinde daha zor pre-generated terrain'e taşı (sende de bu var); Ji tarzı — eğitim adımı arttıkça lineer hız komutunun üst sınırını basitçe büyüt; Margolis tarzı — ödül eşiklerine dayalı grid-genişletme ile hem lineer hem açısal hız aralığını birlikte aç; ya da Li tarzı — hız, DR parametreleri ve ceza terimlerini zorla tek bir on-seviyeli zorluk eksenine sıkıştır.

"Çok fazla manuel tuning yapıyorlar" hissin büyük ihtimalle doğru ve *yapısal* bir nedeni var: bu manuel yöntemlerin ortak sınırı, müfredat tasarımının görev zorluğu ve beklenen performans hakkında öncül bilgiye şiddetle bağımlı olması — eşikler elle ayarlanıyor. Her eşik, her sigmoid takvimi, her "level atlama" kuralı bir hiperparametre; robot değişince, ödül değişince, terrain seti değişince hepsi yeniden ayarlanıyor. Gördüğün deneme-yanılma kültürü, kişilerin tercihi değil, elle-eşikli müfredatın kaçınılmaz bakım maliyeti. Senin pitch'inin en güçlü cümlesi de bu olabilir: "bu tuning döngüsünün kendisini otomatikleştiriyorum."

## LP-ACRL'ye başka ne koyarsın?

Makalenin kendi görev uzayı şu: 5 seviye |vx| (0–2.5 m/s) × 6 seviye |ωz| (0–3.0 rad/s) × 5 terrain tipi × 4 geometri seviyesi = 600 görev örneği (vy dar aralıkta uniform, göreve dahil değil). Yani çekirdek tarif: **komut boyutları + terrain tipi + terrain geometri seviyesi.** Senin eklenti adayların, önceki turnusol testimizden geçenler:

İlk katman (makaleyle bire bir): vx, ωz, terrain tipi, terrain seviyesi. Buna vy'yi de görev boyutu yapmak isteyebilirsin ama makale yapmamış — boyut sayısını şişirmemek için akıllıca; sen de başta dahil etme. İkinci katman (senin katkın olabilecekler): **tepme büyüklüğü frontier'ı** — geçen seferki analizimiz gereği katmanlı kur: küçük-orta tepmeler her env'de DR tabanı olarak kalsın, LP-ACRL'ye ise ayrık birkaç "büyük tepme" seviyesi görev boyutu olarak gir. İstersen **komut geçiş sertliği** (sabit komut vs ani komut değişimleri içeren episodlar) — yüksek hızda dönüş komutuna geçmek başlı başına bir beceri. Üçüncü katmana (yük taşıma görevi, gait parametreleri) şimdilik hiç girme.

Kritik mühendislik uyarısı: görev sayısı çarpımsal patlar. 600 örnek, makalede binlerce paralel env ile besleniyor; her örneğin LP kestirimi için yeterli episoda ihtiyacı var. Tepme frontier'ını eklersen (diyelim 3 seviye) 1800 örneğe çıkarsın — A100'ünde env sayınla bunun oranını hesapla. İlk implementasyonda **makalenin 600'lük uzayını aynen kopyala**, çalıştığını doğrula, boyutu sonra ekle. Bu sana hem debug kolaylığı hem de makaleyle doğrudan karşılaştırılabilirlik verir.

Bir de mimari uyumluluk notu: makalenin politikası **height map gözlemi kullanıyor** (275 boyut) ve gerçek robota LSTM+MLP'li teacher-student distillation ile, gürültülü elevation map'e dayanıklılık için geçiyorlar. Senin hattın ise proprioseptif online-estimation (DreamWaQ/HIM). Bu bir çelişki değil — LP-ACRL müfredat katmanı, gözlem uzayından bağımsız — ama şunu netleştirmen gerek: senin robotunda exteroception var mı, yoksa blind locomotion mı hedef? Blind ise, "LP-ACRL müfredatı + HIM/DreamWaQ politikası" kombinasyonu makalede *olmayan* bir birleşim ve tam senin tezinin ("estimation + iyi env design birbirini tamamlar") test alanı.

## Benchmark nasıl kurulur — makalenin yolu + senin eklemelerin

Makalenin eval iskeleti üç parça, üçü de doğrudan çalınabilir:

**Metrik: EPTE-SP.** Takip hatasını ve stabiliteyi birlikte ölçüyor; düşme anından episode sonuna kadar tüm adımlara en-kötü hata atanıyor, toplam episode uzunluğuna normalize ediliyor. Bunun güzelliği, "iyi takip ama düşüyor" ile "ayakta ama komutu takip etmiyor" hilelerinin ikisini de tek sayıda cezalandırması. Aynen al.

**Başarı tanımı + başarı kümesi.** Bir görev örneği, episode 900+ adım hayatta kalırsa ve hem vx hem ωz için EPTE-SP %30'un altındaysa "başarılı" sayılıyor. Sonra iki eğri raporluyorlar: görev uzayı genelinde **başarı oranı** (kapsam — unmastered alt-uzayı küçültme) ve **başarı kümesi üzerinde ortalama ödül** (ustalık — mastered alt-uzayda performans). Bu iki-amaçlı raporlama senin "hem geniş hem iyi" hikâyeni tam anlatıyor.

**Ayrıştırılmış görselleştirme.** Görev-örnekleme dağılımının zamanla değişim heatmap'leri ve görev-örneği-başına ödül eğrileri — hangi bölgede kimin battığını gösteriyor. Ekip sunumu için altın: "baseline şu hücrelerde hiç öğrenemiyor, biz öğreniyoruz" tek bakışta görünür.

Senin ekleyeceklerin (makalede olmayan, senin tez çizgin): **(a) Sabit held-out eval bataryası.** Makale eğitim görev uzayının kendisinde ölçüyor; sen ek olarak eğitim *sırasında hiç örneklenmeyen* sabit bir test seti tut — hem eğitim aralığı içinden dondurulmuş hücreler hem de hafif OOD hücreler (eğitim üst sınırının %10-20 ötesi hız, görülmemiş terrain-komut kombinasyonları, eğitimde olmayan tepme profilleri). Senin genelleme tezin ancak held-out'ta kanıtlanır. **(b) CVaR raporu.** Her hücrede ortalamanın yanına en-kötü-%10 ortalamasını koy — robustluk iddiasının doğal metriği, ve zaten senin metodolojinde vardı. **(c) Nuisance-DR süpürmesi.** Aynı politikayı sabit görevde, fizik parametrelerinin (sürtünme, yük, gecikme) taranmış değerlerinde değerlendir — estimator'lı mimarinin katkısını gösteren eksen bu. **(d) Çok seed.** Makale gibi en az 3 seed; müfredat yöntemleri seed'e duyarlıdır, tek seed'lik sonuç ekip içinde bile savunulamaz.

Baseline seti de makaleden hazır: Uniform, elle takvim (SC), düşük-ödül-öncelikli (LRPC), ALP ve PLR ile karşılaştırmışlar. Senin için minimum viable karşılaştırma: **mevcut şirket pipeline'ı (Rudin leveler + sabit DR) vs Uniform vs LP-ACRL.** ALP'yi eklemek ucuz (tek satır fark: LP'nin mutlak değeri) ve "işaretli LP neden doğru" ablation'ını bedavaya verir.

## Üst seviye gerçekleştirme planı — karar noktalarıyla

**Faz 0 — Eval altyapısı (önce bu, müfredattan önce).** Sabit eval bataryasını ve EPTE-SP + başarı tanımı + CVaR metriklerini koda dök; mevcut baseline'ı bu bataryada koştur ve sayıları dondur. Karar noktaları: başarı eşiği değerleri (900 adım / %30 makaleden başla, robotuna göre kalibre et); held-out hücre seçimi; OOD sınırının nereye çizileceği; seed sayısı (3 öner). Bu faz bitince elinde "geçilmesi gereken sayı" var — her şey buna karşı ölçülecek.

**Faz 1 — LP-ACRL çekirdeği, makale kopyası.** Algoritma utanç verici derecede basit, bu bir avantaj: her görev örneği için ortalama episodik ödülü izle, LP = ardışık iki değerlendirme arasındaki ortalama ödül değişimi, örnekleme dağılımı = son LP kestirimleri üzerinde sıcaklık parametreli (β) softmax. Karar noktaları: ayrıklaştırma çözünürlüğü (makale: 0.5 m/s ve 0.5 rad/s aralıklar, 4 terrain seviyesi — aynen başla); β sıcaklığı (tek kritik hiperparametre; makale değerini proje sayfasından/koddan bul, yoksa birkaç değerle tara); LP güncelleme periyodu (kaç iterasyonda bir dağıtım tazelenir); env-görev ataması (Isaac Lab'de her env'e reset anında görev örneği atama mekanizması — terrain tipi değişimi env'ler arasında fiziksel terrain grid'ine bağlı, burası entegrasyonun en zahmetli yeri). Nuisance DR bloğuna hiç dokunma — mevcut sabit DR aynen kalsın ki tek değişken müfredat olsun.

**Faz 2 — Kontrollü A/B.** Aynı PPO, aynı ödül, aynı DR, aynı iterasyon bütçesi; sadece müfredat değişiyor: şirket baseline vs Uniform vs LP-ACRL (+ ucuzsa ALP). Faz 0 bataryasında raporla. Karar noktası: "başarı" ilan etme kriterin — önceden yaz (ör. "held-out başarı oranında ≥X puan artış veya aynı başarıya ≤%50 iterasyonda ulaşma"). Önceden yazılmış kriter, sonuç ne çıkarsa çıksın seni hem bilimsel hem politik olarak korur.

**Faz 3 — Senin mimarinle birleşim.** LP-ACRL müfredatının altına HIM/DreamWaQ politikanı koy; "iyi müfredat × online estimation" etkileşimini ölç (2×2: {baseline müfredat, LP-ACRL} × {plain PPO, estimator'lı}). Bu 2×2, senin staj tezinin ("iki ayak birbirini tamamlar") doğrudan deneysel testi ve muhtemelen sunumunun ana tablosu. Karar noktası: blind mi height-map'li mi (yukarıdaki uyumluluk sorusu) — şirket robotunun sensör gerçekliğine göre.

**Faz 4 — Genişletmeler (ancak Faz 2-3 kazanırsa).** Tepme frontier'ını görev boyutu olarak ekle (katmanlı DR tabanıyla); istersen estimator-hatasını fizik-DR örneklemesine sinyal yap (özgün katkı adayın); ADR tarzı sınır genişletmeyi değerlendir. Karar noktası: hangisinin ekleneceği, Faz 2-3'ün *nerede* zayıf kaldığına göre — eval bataryası sana hangi hücrelerin hâlâ kötü olduğunu söyleyecek, genişletmeyi oraya nişanla.

Riskler ve B planları: LP-ACRL sinyali gürültülü çıkarsa (az env/görev oranı) → ayrıklaştırmayı kabalaştır veya LP'ye EMA yumuşatması ekle. β ayarı oynak çıkarsa → makale/proje sayfası değerine sabitle, tarama yapma. Terrain-görev ataması Isaac Lab'de çirkinleşirse → ilk deneyi sadece komut uzayında (düz zemin, makalenin ilk deneyi gibi) yap; o deney tek başına bile "leveler'sız otomatik müfredat çalışıyor" kanıtı verir ve entegrasyonu kademelendirir.

Bir sonraki oturumda istersen Faz 1'in en zahmetli parçasını — Isaac Lab'de görev-örneği atamasının ve LP tamponunun somut sınıf yapısını — birlikte tasarlayalım; proje sayfasında (sites.google.com/view/lp-acrl) implementasyon detayları var deniyor, oradan resmi kod olup olmadığını da kontrol ederek başlarız.


## fable'a sorduğum birkaç güzel soru ve güzel insight

Güzel bilgiler verdin. aklımda birkaç soru var. 

diyelim ki benim şirket şu an rudin style terrain leveler kullanıyor. bu lpacrl, o yöntemden iyi bir sıçrama olur mu?

terrain dışında lpacrl'ye neler verebiliriz? mesela tekmelere karşı düşmemesini (random force miktarı gibi bir parametre düşünüp) ona mı vermek lazım yoksa ikinci dr'ye mi. ya da "beceri" diyebileceğimiz başka neler olabilir. Hmmm. güzel bir yerdeyiz. hadi insight çıkarmaya devam edelim.


Değerlendirdi LP-ACRL'nin avantajlarını ve beceri-nuisance ayrımını sistematikleştirdi
Değerlendirdi LP-ACRL'nin avantajlarını ve beceri-nuisance ayrımını sistematikleştirdi
Şu an tam da işin en verimli yerindeyiz, çünkü ikinci soruyu sorman aslında birinci soruyu yeniden çerçeveliyor. Sırayla.

LP-ACRL, Rudin leveler'dan iyi bir sıçrama mı?
Dürüst cevap: tek başına drop-in replacement olarak düşünürsen, sıçrama mütevazı olabilir — ve nedenini görmen önemli. Rudin leveler zaten kaba bir öğrenme-ilerlemesi curriculum'u; sadece iyi sıralanmış tek bir eksende çalışıyor. Terrain zorluğunun temiz bir sıralaması olduğu için (level 1 → level 10, eğim/basamak artıyor), elle-ayarlı bir merdiven ile ilerleme-tabanlı bir yöntem o eksende neredeyse aynı davranışa yakınsar: ikisi de sınırı yukarı iter. Leveler tam da bu yüzden çalışıyor — sıralaması hazır olan bir eksende.

O zaman LP-ACRL'nin asıl gücü nerede? Sıralamayı gerektirmemesinde. Ve bu, ancak görev uzayını genişlettiğinde işe yarar. Makalenin manşet sonucuna bak: önceki yöntemler ya düz zeminde yüksek hız ya da karmaşık arazide düşük hız yapabiliyordu; LP-ACRL ikisini aynı anda — merdiven/eğim/çakıl/düşük-sürtünmede 2.5 m/s. Buradaki kazanç, terrain ekseninde leveler'ı yenmekten gelmiyor. Kazanç, hız × dönüş × terrain birleşik uzayını tek seferde curriculize edebilmesinden geliyor. "Merdivende yüksek hız", "düz zeminde yüksek hız" ile "merdivende düşük hız"ın toplamı değil; ayrı bir canavar. Bu birleşik zorluğu elle sıralayamazsın — iki bağımsız merdiveni üst üste koyamazsın — ve leveler tam olarak burada duvara toslar.

Yani şirkete satarken doğru çerçeve şu: "LP-ACRL daha iyi bir terrain leveler değil; leveler'ın elle-sıralayamadığı komut-terrain uzayının tamamını birden curriculize etmemizi sağlayan şey — ve bu, leveler'ın erişemediği rejimleri (hızlı + zorlu arazi) açıyor." Sıçrama bu reframe'de. Aksi halde, eğer görev uzayını dar (sadece terrain zorluğu) tutarsan, iyi ayarlanmış bir leveler'a karşı marjinal kalır; o yüzden geçişi kendi X bataryanda A/B'yle ölç, bedava öğle yemeği gibi sunma.

Ve dikkat et — bu seni doğrudan ikinci sorunun neden bu kadar önemli olduğuna götürüyor: "terrain dışında LP-ACRL'ye ne verebilirim?" sorusu, aslında "bu sıçramayı gerçekten nasıl açığa çıkarırım?" sorusunun ta kendisi.

Tekme kuvveti: curriculum mı, DR mı? (sınır vakası, ve öğretici olan da bu)
İlk refleksin muhtemelen şu: "tekme = umursamamasını istediğim şey = DR." Ama dur, daha derin teste vuralım. Geçen sefer koyduğumuz asıl ayrım "yapmasını istediğin" vs "umursamamasını istediğin"di; bunun altındaki mekanik ayrım ise aktif bir davranış edinmek vs bir varyasyona karşı duyarsız olmak.

Tekmeyi bu mercekten geçir: bir şoktan toparlanmak pasif tolerans değil, aktif, öğrenilmiş bir motor beceridir — bir toparlama adımı, bir dengeleyici itiş. Added mass'ten temel farkı bu: ek yükte yürümek için yeni bir "hareket" gerekmez, mevcut yürüyüşünün yüke dayanıklı olması yeter (tolerans). Tekme ise bir reaksiyon talep eder. Üstelik temiz bir zorluk frontier'ı var: küçük tepmeden toparlanmayı, büyükten önce öğrenirsin; itilecek (kelimenin tam anlamıyla) bir sınır var.

Yani tekme büyüklüğü, added mass'ten çok daha fazla bir beceri ekseni — LP-ACRL için iyi bir aday. Bu senin sezginin tersini söylüyor olabilir, ama doğrusu bu.

Buradan çıkan asıl insight ise şu: beceri/robustluk ayrımı parametre başına değil, parametrenin boyutu başına yapılır. Bir tekmenin büyüklüğü skill-difficulty eksenidir (→ curriculum), ama yönü ve zamanlaması birer nuisance'tır (→ geniş DR). Tek bir fiziksel olay ikiye bölünür: zorluk boyutunu curriculum'a, tesadüfi boyutları DR'a verirsin.

Aynı desen sürtünmede de var — ve makalenin kendisi bunun kanıtı. LP-ACRL'nin ustalaştığı araziler arasında "düşük-sürtünmeli düz zemin"i de sayıyorlar. Yani düşük sürtünmeyi arka-plan DR'ına gömmemişler, onu curriculum'a değer bir görev olarak ele almışlar. Mantıklı: buzda yürümek pasif tolerans değil, öğrenilmiş temkinli bir yürüyüştür — sürtünmenin ekstrem ucu skill-benzeridir, orta menzili ise nuisance. Yazarlar sınır vakasını, aktif davranış gerektirdiği için beceri tarafına çözmüşler.

Layering: sınır vakaları için curriculum XOR DR seçmen gerekmez
Tekme için pratik bir tuzak var: büyüklüğü LP ile curriculize edersen ve ajan platoya oturursa, LP tepmelerden çekilir — ama sen robotun deployment boyunca her boy tepmeyi görmesini istiyorsun (robustluk + estimator). Çözüm ikisini üst üste katmak:

Sabit, geniş bir DR "tabanı" her zaman açık kalır (kapsam, robustluk, estimator beslemesi garanti altında), ve LP-ACRL bunun üstünde, sadece zor kuyruğu — büyük tepme frontier'ını — iter. Saf-beceri eksenlerinde (terrain geometrisi, hız) tabana ihtiyacın yok; saf-nuisance eksenlerinde (motor, gürültü) curriculum'a ihtiyacın yok; sınır eksenlerinde (tepme, sürtünme) ikisini birden kullanırsın. Bu, "hepsini tek mekanizmaya basmak" ikilemini tamamen çözer.

Yeniden kullanabileceğin turnusol testi
Herhangi bir parametreyi (ya da parametrenin bir boyutunu) önüne koy ve üç soru sor:

Baş etmek yeni bir aktif davranış mı gerektiriyor, yoksa sadece tolerans mı? Yeni davranış → beceri. Tolerans → nuisance. Aşılan bir zorluk frontier'ı var mı? Var → curriculum kazandırır. Yok (önemi her yerde düz) → DR. Sonunda frontier'da yoğunlaşmasını mı, kalıcı olarak geniş kalmasını mı istiyorum? Frontier → curriculum. Geniş → DR.

Bunu senin eksenlerine uygularsak kabaca şöyle bir harita çıkıyor:

Saf beceri (LP-ACRL'ye): komut hızı, dönüş hızı, terrain geometrisi (basamak yüksekliği, eğim, boşluk genişliği, stepping-stone aralığı), ve varsa zıplama mesafesi/yüksekliği, gövde yüksekliği komutu, komut geçişleri (hızlan-sonra-dön gibi sekanslar). Sınır — layered (curriculum + DR tabanı): tepme büyüklüğü, sürtünmenin ekstrem ucu, ve ilginç bir vaka olarak yük eğer kasıtlı bir taşıma görevi ise (uncontrolled disturbance ise nuisance, ama "şu ağırlığı taşı ve dengeni koru" bir görev hedefiyse beceriye döner — bağlama bağlı). Saf nuisance (DR/ADR'de): motor sürtünme/gain'leri, latency, sensör/gözlem gürültüsü, CoM kayması, kontrolsüz ek kütle.

Dikkat çeken nokta: aynı fiziksel niceliğin (yük, sürtünme, tepme) hangi kovaya düştüğü sabit değil — onu nasıl kullandığına bağlı. Kasıtlı ve frontier'lı mı, yoksa tesadüfi bir bozucu mu? Bu senin gerçek tasarım kararın, ve turnusol testi tam olarak bunu netleştirmek için var.

Bir sonraki adımda istersen şunu somutlaştırabiliriz: bu haritadaki eksenleri legged-gym-Ex'inde tek bir env içinde nasıl yan yana koşturursun — LP-ACRL sampler'ı komut+terrain+tepme-frontier için tek bir birleşik görev vektörü üretirken, ayrı bir DR/ADR bloğu nuisance tabanını beslesin, ve reset anında ikisi nasıl buluşsun. Ya da önce "tepmeyi curriculum'a katmak baseline'ı gerçekten iyileştiriyor mu"yu izole eden minimal bir deney tasarlayabiliriz.

o halde bu lp acrl tarzı bir şey olmadan, mesela hem command uzayını hem de terrain uzayını beceriye katmak istesek nasıl yapardık? önce birinde beceri kazandırıp sonra diğerinde yapsak nasıl olurdu? sıra sıra yapmak ile bunun farkı nedir? bu da bence ilginç ve bilgilendirici bir soru olacak..


Derinlemesine analiz yapılandırıldı
Derinlemesine analiz yapılandırıldı
Bu soru gerçekten öğretici, çünkü "LP-ACRL olmadan iki ekseni birden beceriye nasıl katarız?" diye sorduğun anda, aslında LP-ACRL'nin neden var olduğunu kendi elinle keşfetmiş oluyorsun. Cevabı geometriyle vereyim — çünkü mesele tam anlamıyla geometrik.

İki eksenli bir zorluk ızgarası düşün: yatayda komut zorluğu (hız artıyor), dikeyde terrain zorluğu. Sol-alt köşe kolay-kolay, sağ-üst köşe ise senin asıl hedefin X: hızlı + zorlu arazi. "İki ekseni birden curriculize etmek" demek, bu 2B uzayda sol-alttan sağ-üste bir yol çizmek demek. Bütün mesele hangi yolu izlediğin.

İşte üç strateji, aynı ızgara üzerinde çizdiğim yollar olarak:



Şimdi asıl insight — bu üç yol neden aynı yere farklı maliyetlerle varıyor.

Temel gerçek: beceriler etkileşir, o yüzden çarpım ≠ toplam
"Hızlı + zorlu arazi", "düz zeminde hızlı" ile "zorlu arazide yavaş"ın toplamı değil — ayrı bir canavar. İki ekseni ayrı ayrı ustalaştırıp uç uca ekleyemezsin, çünkü bir eksende öğrendiğin şey diğerinin her ayarına bedava transfer olmuyor. Merdivende koşarken kullandığın yürüyüş, düz zeminde koşarkenkinden farklı; ikisinin kesişimini ayrıca öğrenmen gerekiyor. Sıralı ve ortak yaklaşımların farkı, tam da bu kesişimi ne zaman ve nasıl gördüğünde yatıyor. Izgaradaki yol, o kesişimi ne kadar iyi işlediğini belirliyor.

Bloklu sıralı (kırmızı L) neden en kötüsü — üç ayrı arıza
L yolunun dikey ayağına bak: robot, terrain'i tırmanmaya başladığı andan itibaren sürekli maksimum hızda. Yani terrain curriculum'unun her adımı, aynı anda en zor komut rejiminde yaşanıyor — her basamak devasa bir sıçrama. Bu üç şeyi birden bozar:

Birincisi, hedef köşene (X) en kötü yoldan yaklaşırsın. Senin asıl istediğin sağ-üst köşe; ama L, oraya iki zorluğun bileşkesini tek bir dik ayakta yığarak varır. Küçük, öğrenilebilir artışlar yerine, robotu erkenden bileşik zorluğa çarptırırsın — ve öğrenilecek en kritik bölge (X) en zayıf eğitilen yer olur.

İkincisi, unutma (catastrophic forgetting). Aşama 1'de düz zeminde hızlı yürümeyi öğrenip aşama 2'de bunu bir daha görmezsen, ağ terrain'i öğrenirken o beceriyi silmeye başlar. Sinir ağları, bir beceriyi veri akışında görmeyi bıraktığında onu korumaz. Bloklu yaklaşımda bir bölgeyi "bitirip" geçmek, tam da onu unutmaya davetiye çıkarmaktır.

Üçüncüsü, bir sıra seçmek zorundasın ve doğrusu belli değil. Önce komut mu, önce terrain mi? Cevap bölgeye göre değişebilir (çakılda yavaş öğrenmek iyi, ama merdivende yürüyüş o kadar farklı ki hız erken önem kazanır). Global tek bir sıra buna uyum sağlayamaz.

Dönüşümlü merdiven (mavi) neden birden iyi — ve LP-ACRL'ye kalan fark ne
Merdivene bak: köşegeni kucaklıyor. Her adımda ya biraz komut ya biraz terrain ekleyerek, toplam zorluğu küçük ve öğrenilebilir artışlarla büyütüyor. Uç köşeye erken saldırmıyor, unutma riski düşük (her iki ekseni de sürekli tazeliyor), ve kesişimi yol boyunca görüyor. Yani "sıra sıra azar azar" yapmak, bloklu sıralıdan kategorik olarak üstün — ve aslında ortak yaklaşımın manuel bir yaklaşığı.

İşte can alıcı nokta: merdiven ile LP-ACRL arasındaki fark, "ortak vs ayrı" değil. Merdiven zaten quasi-ortak; köşegeni izliyor. Kalan iki fark şu:

Biri, uyarlanabilirlik. Merdiven sabit bir takvimle ilerler — robotun gerçekte nerede zorlandığına bakmaksızın her turda bir çentik komut, bir çentik terrain. LP-ACRL ise ilerlemenin fiilen olduğu yere gider: bir bölgede takılırsa orada oyalanır, bir bölge çözülmüşse atlar. Cephe, takvime değil, robotun anlık yeterliğine göre bükülür.

Diğeri, serbest köşegen adımları. Merdiven yalnızca eksen-hizalı hareket edebilir (ya sağa ya yukarı). LP-ACRL "biraz daha hızlı ve biraz daha pürüzlü" gibi gerçek bir köşegen adımı örnekleyebilir — ki en öğrenilebilir sonraki görev çoğu zaman tam da budur, ve hiçbir eksen-hizalı müfredat bunu öneremez.

Yani LP-ACRL'nin manuel iyi bir müfredata kattığı asıl değer, temiz bir şekilde izole oluyor: uyarlanan cephe + serbest köşegen. "Ortak yapmak"ın faydasını zaten merdivenle alıyorsun; LP bunun üstüne adaptasyonu koyuyor.

Peki LP-ACRL olmadan ortak yolu manuel çizsek?
Tam da senin sorunun kalbi burada. Ortak (köşegen) müfredatı elle kurmak istersen, komut ve terrain'i tek bir birleşik zorluk skalarında birleştirip onu rampalaman gerekir — yani "bu görev ne kadar zor?"u tek sayıya indiren bir formül yazman lazım. Ama bu formülü yazmak zor, çünkü etkileşim doğrusal değil: "orta hız + orta terrain"in zorluğu, ikisinin ayrı zorluklarının toplamı değil. Sıralı yaklaşım, insanların bu birleşik-zorluk-tanımlama sıkıntısından kaçmak için başvurduğu numaradır — ama bu kaçışın bedeli, yukarıdaki üç arıza (unutma, uç köşe, sıra seçimi).

LP-ACRL'nin tek cümlelik varlık nedeni işte bu: birleşik zorluğu tanımlamak zorunda kalmadan, ortak müfredatın faydasını almanı sağlar. Zorluğu sen sıralamıyorsun; robotun ilerlemesi onu ölçüyor. Merdiven de bu tanımdan kaçar (eksen-eksen ilerleyerek), ama sabit takvimle; LP ölçülen ilerlemeyle.

Ne zaman sıralı gerçekten meşru?
İki durumda:

Eksenler ayrıştırılabilirse — yani bir eksendeki ustalık diğerinin tüm ayarlarına tam transfer oluyorsa — sıralı hem doğru hem ucuzdur; kesişim diye bir canavar yoktur. Ama lokomosyonda eksenler ayrıştırılamaz; "hızlı + zorlu"nun ayrı bir beceri olması bunun kanıtı. Bu yüzden lokomosyonda saf sıralı nadiren doğru araçtır.

Bir de bootstrap/warm-start olarak meşrudur: robot düz zeminde temel yürüyüşü kapmadan tüm çarpım uzayını basmak erken kaosa yol açar. Kısa bir "önce düz zeminde ayakta dur ve yürü" fazı, ortak müfredat için iyi bir başlangıç noktası verir. Gerçek pipeline'lar da bunu yapar: tam bloklu ustalık değil, kısa bir tohumlama, sonra çarpım uzayında ortak müfredat.

Pratik reçete bu yüzden: kolay köşede kısa bir bootstrap, ardından çarpım uzayında ortak (tercihen uyarlanan) müfredat — bloklu tam-ustalık değil.

Bir sonraki adımda istersen bu "birleşik zorluk skalarını yazamama" sorununu somutlaştırıp, LP-ACRL'nin ilerlemeyi tam olarak hangi büyüklükten (episodik ödül eğimi mi, başarı oranı değişimi mi) kestirdiğine ve bunu senin legged-gym-Ex'inde hangi sinyalden okuyacağına inebiliriz — çünkü "ilerleme"yi nasıl ölçtüğün, bu cephenin ne kadar temiz büküleceğini belirleyen asıl mühendislik kararı.


## notasyon 

C: Simülatörde değiştirebildiğimiz her şey.
A: C içinden eğitimde kullanmayı seçtiğimiz parametreler.
B: Seçilen parametrelerin değer aralıkları.
gamma: eğitimde ajana göstereceğim enviromentlar (bulmak istediğim şey!) "Şimdi ajana hangi parametreli dünyayı, environmentı göstereyim?"
X: hedeflerim: "Robot şöyle şöyle ortamlarda, şöyle şöyle etkiler altında bile düşmeden başarıyla gidebilsin"