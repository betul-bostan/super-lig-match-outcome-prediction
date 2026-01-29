# ⚽ Süper Lig Skor Tahminleme ve Algoritma Performans Analizi

Bu proje, makine öğrenmesi modellerinin spor müsabakaları gibi yüksek varyanslı alanlardaki başarımını test etmek amacıyla geliştirilmiştir. 

## 📊 Model Performans Analizi
Proje kapsamında 5 farklı algoritma yarıştırılmış ve en yüksek başarı oranı **%52.94 ile Random Forest** modelinde gözlemlenmiştir.

### Teknik Karşılaştırma Sonuçları:
* **Random Forest:** %52.94
* **SVM:** %52.84
* **Logistic Regression:** %52.65
* **ANN (Yapay Sinir Ağları):** %49.04

## 📉 Neden %52? (Veri Kısıtlılığı Analizi)
Elde edilen bu oran, bir "başarısızlık" değil, mevcut veri setinin bilgi kapasitesinin (information gain) sınırıdır. Performansın bu seviyede kalmasının teknik nedenleri şunlardır:

* **Eksik Öznitelikler (External Features):** Model sadece geçmiş skorlara odaklanmaktadır. Maç sonucunu doğrudan etkileyen; sakatlıklar, hava durumu, hakem eğilimleri ve takımların o haftaki psikolojik durumları gibi kritik dışsal veriler veri setinde bulunmamaktadır.
* **Yüksek Varyans:** Sporun doğasındaki sürpriz faktörü, eldeki sınırlı veriyle ulaşılabilecek en istikrarlı **benchmark (temel)** değerlere ulaşılmasına neden olmuştur.
* **Mühendislik Çıkarımı:** Bu çalışma, veri madenciliğindeki "Garbage In, Garbage Out" prensibini doğrulamakta ve daha geniş bir veri seti için bir temel oluşturmaktadır.
