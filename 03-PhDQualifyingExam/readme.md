# PhD Qualifying Exam - Doktora Yeterlilik Sınavı (29 Nisan 2026)

## Soru 1
**Soru:** 2. dereceden verilen denklemi newton yöntemi ile 2 iterasyon yaparak çözünüz. Her iterasyon sonucunda loss değerini hesaplayınız.
* Başlangıç noktası olarak x0=[3 2] alın. 
* alpha = 0.1 olarak alın.

**Cevap:**

* Normalde alpha vermemiş olsaydı direkt 2. dereceden denklemi klasik newton ile tek iterasyonda sonuca ulaştırırdık.
* g(x): x1 ve x2 ye göre kısmi türevleri alarak gradienti buluyoruz.
* H(x): x1 ve x2 ye göre ikinci türevleri alarak hessian matrisini buluyoruz.
* H^-1(x): hessian matrisinin tersini alarak hessianın tersini buluyoruz.
* Klasik NewtonDenklem: **xi+1 = xi - H^-1(xi) * g(xi)**
* Ama alpha verildiği için: **xi+1 = xi - alpha * H^-1(xi) * g(xi)** şeklinde denklemi güncellememiz gerekiyor.
* Bu durumda ilk ve ikinci iterasyon sonucunda oluşan x1 ve x2 değerleri ile gitmemiz gereken final değerine doğru adım adım ilerlemiş oluyoruz.
* Soruda alpha sız sonucu hesaplayarak gidilmesi gereken yerin ne olduğunu yazdım.
* Sonrasında alpha olan durumda 2 iterasyon sonucunda hangi noktalara geldiğimizi ve bu noktalarda loss değerlerini hesapladım.
* Loss değeri için **x0 - x1** ve **x1 - x2** arasındaki farkların normunu aldım.
