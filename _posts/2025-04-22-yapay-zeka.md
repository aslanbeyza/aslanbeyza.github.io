---
title: "Büyük Dil Modelleri ile Generative AI"
date: 2025-04-23 12:00:00 +0300
categories: [Büyük Dil Modelleri ile Generative AI]
tags: [kişisel, giriş]
---

## TRANSFORMER

Transformerı anlamak için [**encoder**] ve [**decoder**]  , [**Dikkat Mekanizması**]  , [**Transfer Learning teknikleri**]ni  anlamak gerekir. Hemen Encoder-Decoder ile başlayalım.

### Encoder 
Transformerdan önce LSTM gibi RNN gibi mimariler doğa dil işlemede çok önemliydi.Bu mimarilerin en büyük özelliği ağ bağlantılarına geri bildirimler sağlamasıydı.


![Desktop View](/assets/img/RNNYapısı.png){: .w-75 .mx-auto d-block }


Ekranda bir RNN Mimarisini görüyorsunuz bu RNN hücresini açtığınızda bu şekilde bir yapı sizi karşılar.

Dikkat ederseniz 
 
 - 1. girdi ilk hücreye giriyor ve bu hücreden bir çıktı elde ediliyor.
 - 2. Daha sonra ikinci girdi ile beraber birinci girdinin çıktısı hücreye giriyor. Her hücreye bir önceki hücrenin çıktısıda veriliyor.Buna **Hidden State** denir.Her bir hücrede bir önceki çıktıda kullanıldığı için ağfa bir hafıza oluşuyor. bu RNN Mimarileri zaman serileri, ses işleme gibi doğal dil işleme görevlerinde sık sık  kullanıldı. RNN mimarileri ile metinde üretebilirsiniz.

![Desktop View](/assets/img/EncoderDecoder.png){: .w-75 .mx-auto d-block }


RNN Mimarileri Encoder, Decoder veya Seq2seq  görevler içinde kullanıldı. Bu görevlerde modele bir dizi veriliyor ve çıktı olarak  bir dizi elde edilmek isteniyor.

**Encoder** girdi dizilerini sayısal temsillere çevirir.
**Decoder** Bu veriler decoder kısmına verilir. Decoder ise çıktı dizisini üretir.


![Desktop View](/assets/img/EncoderDecoderArray.png){: .w-75 .mx-auto d-block }


Yandaki resimde Transformers are great! cümlesinin almancası mevcut resimdede görüldüğü üzere modele bir dizi giriyor ve bir dizi çıkıyor.
Encoder blok bu girdi dizilerini sayısal temsillere çeviriyor. State kısmını bir zaman adımında ortaya çıkan veri olarak düşünebilirsiniz.Son statedeki bilgiler decoder bloğuna geçiyor ve bu blokta bir dizi üretiliyor.Bu mimarinin zayıf tarafı encoder bloğundaki son gizli state kısmına bir dar boğazın oluşması uzun bir metin düşünün ve bu metindeki tüm kelimeler işleniyor.Bu kelimeler işlene işlene sonlara doğru bu veriler birikiyor ve en son state decoder bloğuna geçerken bir dar boğaz oluşuyor işte bu sorunu çözmek için [**Attention Mekanizması**] kullanılıyor. Yani Dikkat Yaklaşımı mekanizması gerçekleştirildi.
![Desktop View](/assets/img/attention.png){: .w-75 .mx-auto d-block }


Bu dikkat mekanizması decoder kısmının encoder kısmındaki tüm statelere ulaşmasını sağlıyor.Bu Dikkat Mekanizması Modern Yapay Zeka Mimarisinde kilit rol oynadı.Dikkat Mekanizmasının arkasında yatan ana fikir gizli state kullanımı. Encoder sondaki gizli state kısmını decoder kısmına veriyordu.Dikkat mekanizmasında her bir adım için bir gizli state üretir.Her bir RNN hücresi bir state çıktılıyor ama aynı zaman adımındaki tüm stateler decoder için büyük bir girdi oluşturabilir.Dolayısıyla bazı mekanizmaların hangi statelerin daha önemli olduğunu belirlemesi gerekir.İşte tam burada attention yani dikkat mekanizması devreye girer 
![Desktop View](/assets/img/attention2.png){: .w-75 .mx-auto d-block }


Bu mekanizma her bir state için farklı büyüklükteki ağırlıklar atar. Böylece dar boğaz sorunu çözülmüş olur.

💕 Şimdi ise attention ı anlamak için bir çeviri görevini ele alalım  
![Desktop View](/assets/img/çevirigörevi.png){: .w-75 .mx-auto d-block }


Burada ingilizce bir cümle fransızcaya çevrilmiş burada her bir piksel bir ağırlığı temsil eder.Attention kelimeler arasındaki ilişkiyi öğrenir ve hangi kelimeler arasında ilişki var hangi kelimeler arasında ilişki yok olduğunu belirler.Buna göre ağırlıkları atar. İngilizce cümlede dikkat edin bir area keilmesi mevcut frasızcada bu kelimeye karşılık zone kelimesi geliyor.Bu iki kelimenin cümledeki yerleri farklı dimi biri 6. sırada diğeri 5. sırada işte attention bu iki kelimenin ilişkili olduğunu öğrenir.Ancak şöyle eksiklikler var biz modele dizi veriyoruz dimi ve çıktı olarak dizi alıyoruz Bu dizideki kelimeler tek tek işleniyor.İşte bu işleme sırasında hesaplamalar sırayla yapılıyor.yani bir işlem yapılıyor o bitiyor sonra diğer işlem yapılıyor.İşte bu işlemler paralel şekilde yapılamıyor.Bu problemin çözülebilmesi içinde transformer mimarisi geliştiriliyor.
![Desktop View](/assets/img/Transformer.png){: .w-75 .mx-auto d-block }


Ekranda gördüğünüz gibi bu mimari attention yerine self attention yaklaşımını kullandı.Normalde sinir ağları katmanlardan oluşur dimi bu katmanlarda nöronlar yani birimler vardır.Attention her bir birimdeki statelerle çalışıyordu.Self attention ise aynı katmanda bulunan tüm birimlerdeki stateler üzerinde çalışır.
Resimdede görüldüğü gibi hem encoder hem decoder kendi self attention mekanizması var. Bu mekanizmalarda RNN lerin kullanıldığı tekrarlı sinir ağı yerine ileri beslemeli sinir ağı kullanıldı.Bu mimari RNN mimarisinden daha hızlı eğitildi.Bu teknik Fatih in İstanbulu fethi gibi bir çağ açtı bir çağ kapattı.Bu mimari kullanılarak doğal dil işlemede birçok model geliştirildi.Orijinal transformer mimarisi büyük miktarda veri ile çok güçlü bilgisayarlar kullanılarak sıfırdan eğitildi . Ancak birçok doğal dil işleme uygulaması için pratikte etiketli veri bulmak oldukça zordur.Ayrıca bu modelleri eğitmek için güçlü bilgisayarlar gereklidir.
Çoğu kişide maalesef böyle bilgisayarlar yok.İşte bu sorunları çözmek için[**Transfer Learning**] yaklaşımı geliştirildi. 


![Desktop View](/assets/img/TransferLearning.png){: .w-75 .mx-auto d-block }


Transfer Learning kısaca daha önceden eğitilmiş bir modeli alıp onu kendi projenize adapte etmenizdir.Sıfırdan model eğitmemize gerek yok örneğin ResNet gibi bir sinir ağını alıp kendi veriniz için onu fine-tuning edebilirsiniz.


![Desktop View](/assets/img/finetuning.png){: .w-75 .mx-auto d-block }


Büyük dil modellerinde fine-tune kavramı çok sık duyulur.Bu teknik eğitilmiş bir modelin hiperparametreleri ile oynamak olarak düşünebilirsin.Yani bir gitarı akort etmek gibi 
![Desktop View](/assets/img/SupervisedLearning.png){: .w-75 .mx-auto d-block }


Eskiden A alanındaki bir problem için ayrı bir model üretilirdi B alanındaki bir problem içinde başka bir model eğitilirdi.Her problem için ayrı ayrı modeller eğitilirdi.
![Desktop View](/assets/img/TransferLearning2.png){: .w-75 .mx-auto d-block }


Transfer Learning tekniği ile mimariler gövde ve baş olarak ayrıldı. Bir proje yaparken gövdeyi aynen alırsınız baş kısmını probleminize göre ayarlarsınız. Böylece A alanındaki eğitilen bir görev için modeli B alanı içinde kullanabilirsiniz.Özellikle Computer Vision alanında Transfer Learning kullanıldı.


Modeller ilk olarak 

![Desktop View](/assets/img/Animals.png){: .w-75 .mx-auto d-block }


Modeller ilk olarak imageNet gibi büyük veri setlerinde eğitildi Bu sürece pre-training yani ön eğitim denilir.Bu eğitimde  model, köşe, renk gibi temel özellikleri öğrenir.Örneğin köşeler her resimde var dimi bunlar temel feature yani öz niteliklerdir.İşte bu ön eğitimli modeli fine-tune edererk çiçek sınıflandırma gibi görevler içinde kullanabiliriz.


![Desktop View](/assets/img/resnet.png){: width="972" height="589" .w-75 .normal}


Örneğin kedi köpek sınıflandırma gibi bir probleminiz var  


![Desktop View](/assets/img/resnet2.png){: .w-75 .mx-auto d-block}


işte bu problemi çözmek için resNetin son katmanını kendi probleminiz için ayarlamanız yeterli Biraz ilginç gelecek ama bu Transfer Learning kullanılarak inşa edilen modeller sıfırdan eğitilen modellerden daha iyi performans gösterdi.

![Desktop View](/assets/img/ConputerVision.png){: width="972" height="589" .w-75 .normal}


Transfer Learning Computer Vision yani Bilgisayar görüsü alanındaki görevler için başarılı bir şekilde uygulandı.Fakat 2017 yılından önce bu teknik doğal dil işleme için tam uygulamanadı.O zamanlar doğal dil işleme projeleri yapmak için büyük miktarda etiketli veri gerekiyordu.Büyük miktarda veriniz olsa bile Computer Visiondaki modellerin başarısı Nlp görevleri için elde edilemiyordu.


![Desktop View](/assets/img/Bert.png){: width="972" height="589" .w-75 .normal}


2017 yılından sonra yeni bir yaklaşımla transfer learning  doğal dil işleme içinde kullanılabilir hale getirildi.Bu denetimsiz ön işleme ile elde edilen feature lar kullanılarak yapıldı.Bu teknikle metin sınıflandırma problemleri yüksek doğrulukla çözüldü. 



![Desktop View](/assets/img/user.png){: width="972" height="589" .w-75 .normal}


Daha sonra bu tekniği uygulamak için ***ULMFİT*** kütüphanesi geliştirildi.Bu kütüphane transfer learningi 3 aşama ile uyguladı.

- 1. aşamada **pretraning** yapıldı.Bu adımda gelecek kelimelerin tahmini önceki kelimelere dayanarak yapıldı bu işleme dil modelleme denildi.Bu yaklaşımda etiketli veriye ihtiyaç duyulmadı.Yani wikipedia gibi kaynaklardan bolca yararlanıldı.


- 2. **Adaptasyon**  diyelim ki modeli wikipedia üzerinde eğittiniz Bu modeli film yorumları içeremn IMDB ye adapte edebilirsiniz.
Bu aşamada dil modellemeyi kullanıyor.Fakat model hedef deki gelecek kelimeyi tahmin eder.


- 3. **Fine Tuning** son aşama yani ince ayar dil modeli bir görev için sınıflandırma katmanı ile Fine-tuning edilir. ULMFİT kütüphanesi Transfer Learning ve preTraining tekniklerini doğal dil işlemede kullanılmasını sağladı.


![Desktop View](/assets/img/EvolutionofLLM.png){: width="972" height="589" .w-75 .normal}


2018 de Transfer Learning ile self attention ı birleştiren 2 önemli transfer learning geliştirildi.Bunlar Gpt ve Bert modelleriydi.


![Desktop View](/assets/img/gptbert.png){: .w-75 .mx-auto d-block }


gpt transformer mimarisinin decoder bloğunu kullanırken Bert transformer mimarisinin encoder bloğunu kullandı.Bert maskelenmiş dil modelidir.Bu modeller bir metindeki maskelenmiş rastgele kelimeleri tahmin eder.


![Desktop View](/assets/img/sentence.png){: .w-75 .mx-auto d-block }


işte model cümledeki mask ile kapatılmış kelimeleri tahmin eder.Bert ve gpt ile yeni bir transformer çağı başlatıldı.


![Desktop View](/assets/img/Huggingface.png){: width="972" height="589" .w-75 .normal}


işte Hugging Face tarafından geliştirilen Transformers kütüphanesi ile büyük Dil Mpodelleri için bir standart getirildi. Bu kütüphane 


![Desktop View](/assets/img/framework.png){: width="972" height="589" .w-75 .normal}


resimdeki frameworkleri desteklemektedir.Bunun anlamı Transformerlarla çalışmak için bu 3 frameworküde kullanabilirsiniz.


![Desktop View](/assets/img/sorucevap.png){: width="972" height="589" .w-75 .normal}


Ayrıca transformers göreve özgü mimarilerde sundu. Yani metin sınıflandırma veya soru cevap gibi projelerizi yapmak için bu mimarilerin baş tarafını fine-tune etmek yeterli olacaktır.Böylece kısa zamanda mdoelinizi eğitip probleminizi çözebilirsiniiz.
Kısaca Büyük dil modelleri ile yapılan çalışmalar hızlandı ve gerçek hayat problemlerinin çözümü kolaylaştı.



