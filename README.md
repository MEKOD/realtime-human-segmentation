🫥 Realtime Cloak

Gerçek zamanlı insan segmentasyonu ile “görünmezlik” efekti.
Green screen yok. Render yok. Offline kurgu yok.
Kamera açıkken, canlı.

Bu proje; bilgisayarlı görü, arka plan modelleme ve gerçek zamanlı görüntü işleme kullanarak kişinin kendisini canlı olarak arka planla harmanlamasını sağlar.

🚀 Özellikler

🎥 Gerçek zamanlı çalışır (kamera açıkken)

🧠 AI tabanlı insan segmentasyonu (MediaPipe)

🌫️ Yarı saydam / hayalet modu

🖼️ Dinamik arka plan öğrenme (background modeling)

🪶 Kenar yumuşatma & maske stabilizasyonu

🧪 Green screen gerektirmez

⚡ Render yok, bekleme yok

🧠 Nasıl Çalışır?

Kamera açılır

Sistem ortamın arka planını kısa sürede öğrenir

İnsan, AI ile canlı olarak segment edilir

İnsan pikselleri arka planla harmanlanır

Sonuç: Gerçek zamanlı “cloak / görünmezlik” efekti

Bu bir video efekti değil, canlı bilgisayarlı görü uygulamasıdır.

⌨️ Kontroller
Tuş	İşlev
ESC	Çıkış
r	Arka planı yeniden öğren
c	Cloak (görünmezlik) aç / kapat
🛠️ Kurulum
Gereksinimler

Python 3.9 – 3.11

Kamera (webcam)

Kurulum
pip install opencv-python mediapipe numpy

Çalıştırma
python gorunmez.py

🧪 Kullanılan Teknolojiler

Python

OpenCV

MediaPipe (Selfie Segmentation)

NumPy

⚠️ Notlar

Düşük ışıkta segmentasyon kalitesi düşebilir

Sabit arka plan, daha iyi sonuç verir

Gerçek zamanlı olduğu için donanıma duyarlıdır

🎯 Amaç

Bu proje;

Computer Vision pratiği

Gerçek zamanlı görüntü işleme

Segmentasyon + compositing mantığını göstermek

amacıyla geliştirilmiştir.
