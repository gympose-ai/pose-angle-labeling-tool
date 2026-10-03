# Eklem Acisi Hesaplama Tanimi

Bu dokuman, canvas uzerinde gosterilen acilarin hangi eklem noktalarindan ve hangi vektorlerden hesaplandigini tanimlar. Amac, aci ozelliklerinin hem yazilim icinde izlenebilir olmasi hem de literaturde yaygin kullanilan iskelet-tabanli hareket analizi yaklasimlariyla uyumunun aciklanmasidir.

## Hesaplama Yontemi

Her aci uc keypoint ile tanimlanir:

```text
A - B - C
```

Burada `B` merkez/tepe eklemdir. Aci, `B -> A` ve `B -> C` vektorleri arasinda hesaplanir.

```text
v1 = A - B
v2 = C - B
aci = arccos((v1 . v2) / (|v1| |v2|)) * 180 / pi
```

Sistemde bu hesaplama `pose_editor.py` icindeki `calcAngle(a, b, c)` fonksiyonuyla yapilir. Fonksiyonun dondurdugu deger derece cinsindendir.

## COCO-25 Keypoint Indeksleri

| Indeks | Keypoint |
|---:|---|
| 5 | neck |
| 6 | left_shoulder |
| 7 | right_shoulder |
| 8 | left_elbow |
| 9 | right_elbow |
| 10 | left_wrist |
| 11 | right_wrist |
| 12 | left_hip |
| 13 | right_hip |
| 14 | hip |
| 15 | left_knee |
| 16 | right_knee |
| 17 | left_ankle |
| 18 | right_ankle |
| 19 | left_big_toe |
| 21 | left_heel |
| 22 | right_big_toe |
| 24 | right_heel |

## Standart Aci Tanimlari

| Arayuz etiketi | Uc keypoint `(A, B, C)` | Merkez eklem | Hesaplanan vektorler |
|---|---|---|---|
| R.Omuz | `(right_elbow, right_shoulder, right_hip)` = `(9, 7, 13)` | right_shoulder | right_shoulder -> right_elbow ve right_shoulder -> right_hip |
| L.Omuz | `(left_elbow, left_shoulder, left_hip)` = `(8, 6, 12)` | left_shoulder | left_shoulder -> left_elbow ve left_shoulder -> left_hip |
| R.Dirsek | `(right_shoulder, right_elbow, right_wrist)` = `(7, 9, 11)` | right_elbow | right_elbow -> right_shoulder ve right_elbow -> right_wrist |
| L.Dirsek | `(left_shoulder, left_elbow, left_wrist)` = `(6, 8, 10)` | left_elbow | left_elbow -> left_shoulder ve left_elbow -> left_wrist |
| R.Kalca | `(right_shoulder, right_hip, right_knee)` = `(7, 13, 16)` | right_hip | right_hip -> right_shoulder ve right_hip -> right_knee |
| L.Kalca | `(left_shoulder, left_hip, left_knee)` = `(6, 12, 15)` | left_hip | left_hip -> left_shoulder ve left_hip -> left_knee |
| R.Diz | `(right_hip, right_knee, right_ankle)` = `(13, 16, 18)` | right_knee | right_knee -> right_hip ve right_knee -> right_ankle |
| L.Diz | `(left_hip, left_knee, left_ankle)` = `(12, 15, 17)` | left_knee | left_knee -> left_hip ve left_knee -> left_ankle |
| R.AyakBilegi | `(right_knee, right_ankle, right_big_toe)` = `(16, 18, 22)` | right_ankle | right_ankle -> right_knee ve right_ankle -> right_big_toe |
| L.AyakBilegi | `(left_knee, left_ankle, left_big_toe)` = `(15, 17, 19)` | left_ankle | left_ankle -> left_knee ve left_ankle -> left_big_toe |
| R.AyakYonu | `(right_heel, right_ankle, right_big_toe)` = `(24, 18, 22)` | right_ankle | right_ankle -> right_heel ve right_ankle -> right_big_toe |
| L.AyakYonu | `(left_heel, left_ankle, left_big_toe)` = `(21, 17, 19)` | left_ankle | left_ankle -> left_heel ve left_ankle -> left_big_toe |

## Turetilmis Teknik Metrikler

Uzman degerlendirmesinde gerekli gorulen bazi degerler klasik `A-B-C` eklem acisindan farklidir. Bu nedenle sistem, standart acilara ek olarak `derived_metrics` alaninda asagidaki 2D teknik metrikleri de kaydeder.

| Etiket | Hesaplama | Referans |
|---|---|---|
| GovdeSapma | Govde hattinin dikey eksenden sapmasi. Omuz merkezi icin `neck`; kalca merkezi icin `hip` kullanilir. Bu noktalar yoksa sag/sol omuz veya sag/sol kalca orta noktasi kullanilir. | Dikey eksen |
| R.KolSapma / L.KolSapma | Omuzdan bilege kol hattinin yatay eksenden sapmasi. Bilek yoksa dirsek kullanilir. | Yatay eksen |
| R.BacakSapma / L.BacakSapma | Kalcadan ayak bilegine bacak hattinin yatay eksenden sapmasi. Ayak bilegi yoksa diz kullanilir. | Yatay eksen |
| R.KolGeriGidis / L.KolGeriGidis | Omuz merkezli yonlu kol acisi. Referans `shoulder -> hip` govde hattidir; hedef `shoulder -> wrist` kol hattidir. Bilek yoksa dirsek kullanilir. Deger 0-360 derece araligindadir. | Govde referansi |
| R.KalcaFleksExt / L.KalcaFleksExt | Mevcut kalca acisinin fleksiyon/ekstansiyon amacli etiketlenmis kopyasi. | Omuz-kalca-diz |
| R.DizEkst / L.DizEkst | Mevcut diz acisinin ekstansiyon amacli etiketlenmis kopyasi. | Kalca-diz-ayak bilegi |
| Iki bacak arasi aci | Sag ve sol bacak uclarinin kalca merkezinde olusturdugu aci. Ayak bilekleri tercih edilir; yoksa dizler kullanilir. | Kalca merkezi |
| Pelvis acisi | Sag-sol kalca hattinin yatay eksenden sapmasi. | Yatay eksen |

PDF/uzman mesajlarinda istenen `Bacak acisi`, `Govde acisi`, `Diz fleksiyon acisi`, `Iki bacak arasi aci`, `Govde-bacak acisi`, `Kollarin yatay acisi`, `Parabolik/ucus acisi`, `Kalca-govde eksantisyonu`, `Bacak yatay sapmasi`, `Pelvis acisi`, `Kollarin yanda acisi`, `Govde kalca fleksiyonu`, `Kalca ekstansiyonu`, `Kollarin geriye gidisi`, `Govde Sapmasi`, `Kol Sapmasi`, `Bacak Sapmasi` ve `Diz ekstansiyonu` adlari arayuzde ayrica listelenir. Bu adlar mevcut kisa teknik etiketleri silmeden ilgili 2D metriklere baglanir. `Parabolik/ucus acisi` tek karede gercek ucus parabolu yerine 2D tek-kare gostergesi olarak kaydedilir; fiziksel ucus acisi icin coklu frame takibi gerekir.

Bu metrikler ekranda `Acilar` modu acikken turuncu etiketlerle gosterilir. `Apply & Save` sonrasinda hem ana JSON icindeki `derived_metrics` alanina hem de ayri `_angles.json` dosyasina yazilir. Genel `angles` listesi artik `standard_angles + derived_metrics + manual_angles` siralamasiyla uretilir.

`Aci Sec` modu acikken kullanici yalnizca secmek istedigi standart acilari ve teknik metrikleri listeden isaretleyebilir. Bu durumda `Apply & Save`, tum standart/turetilmis metrikler yerine secilen kayitlari yazar; eski `Acilar` modu ise tumunu kaydetmeye devam eder.

## Ozel Aci Modu

Arayuzdeki `Ozel Aci` modu, standart tabloda olmayan acilar icin kullanilir. Kullanici uc keypoint secer:

```text
1. nokta -> merkez nokta -> 3. nokta
```

Bu secimde ikinci keypoint merkez kabul edilir ve aci yine ayni `A - B - C` formuluyle hesaplanir. Boylece standart aci listesinde olmayan ancak uzmanin incelemek istedigi herhangi bir anatomik veya teknik hareket acisi goruntulenebilir.

## Aci Silme Modu

Arayuzdeki `Aci Sil` butonu secili acilari kayittan cikarmak icin kullanilir. Buton aktifken silinmek istenen standart veya ozel aci etiketine/yayina tiklanir. Standart acilar `deleted_standard_angles` alaninda etiket bazli saklanir ve sonraki kayitlarda `standard_angles` ile toplam `angles` listesine eklenmez. Ozel acilar ise `manual_angles` listesinden kaldirilir.

`Apply & Save` islemiyle ekranda gorunen acilar kaynak JSON dosyasina kaydedilir. Standart `Acilar` gorunumu aciksa bu kayitlar `standard_angles` alanina, manuel secilen ozel acilar `manual_angles` alanina yazilir. Tum acilar birlikte `angles` alaninda da tutulur. Ayni kayitlar ayrica `easy_ViTPose/temp/açılar/` klasorune `_angles.json` uzantili ayri bir dosya olarak yazilir. Her kayitta secilen keypoint indeksleri, keypoint adlari, merkez eklem, iki vektorun yonu, acinin derece karsiligi ve hesaplama yontemi bulunur. Bu sayede otomatik standart listede olmayan acilar da veri setine eklenebilir.

Ornek:

```json
{
  "manual_angles": [
    {
      "label": "manual_angle_1",
      "keypoint_indices": [6, 8, 10],
      "keypoint_names": ["right_shoulder", "right_elbow", "right_wrist"],
      "vertex_index": 8,
      "vertex_name": "right_elbow",
      "angle_degrees": 145.32,
      "source": "manual_canvas_selection"
    }
  ]
}
```

## Literaturle Uyum

Literaturde iskelet-tabanli hareket tanimada eklem koordinatlari, kemik/segment vektorleri, eklem acilari ve zamansal degisimleri yaygin ozellikler olarak kullanilir. Bu projedeki yontem, iki komsu segment vektoru arasindaki aciyi hesaplayan klasik geometrik yaklasimdir.

Bu yaklasim su calismalarla uyumludur:

- Ohn-Bar ve Trivedi (CVPR Workshops 2013), iskelet takibinden turetilen eklem acisi ozelliklerini eylem tanima icin kullanmistir.
- Chen vd. (Electronics 2023), OpenPose benzeri keypointlerden eklem acisi, eklem mesafesi ve hiz gibi geometrik/temporal ozellikler cikarmistir; aciyi uc eklem noktasinin olusturdugu geometri uzerinden tanimlar.
- Son donem ergonomi ve pose-estimation calismalarinda, ozellikle dusuk serbestlik dereceli diz ve dirsek gibi eklemler icin aci, proksimal ve distal segment vektorleri arasindaki aci olarak hesaplanmaktadir.

## Kaynaklar

1. Ohn-Bar, E., & Trivedi, M. M. (2013). Joint Angles Similarities and HOG2 for Action Recognition. CVPR Workshops. https://openaccess.thecvf.com/content_cvpr_workshops_2013/W12/html/Ohn-Bar_Joint_Angles_Similarities_2013_CVPR_paper.html
2. Chen, Y. et al. (2023). Human Action Recognition Based on Skeleton Information and Multi-Feature Fusion. Electronics, 12(17), 3702. https://doi.org/10.3390/electronics12173702
3. Barker, R., & Lee, S. H. (2026). 3D human pose keypoints and corresponding joint angle calculation for vision-based WMSD risk assessments. Ergonomics. https://doi.org/10.1080/00140139.2026.2639614
4. Wu, S., Li, H., Qiao, S., & Wang, G. (2026). Branch-Splitter multi-granularity feature fusion for local joint-angle estimation. Pattern Recognition, 172, 112578. https://doi.org/10.1016/j.patcog.2025.112578

## Sinirlar

Bu sistem 2D goruntu koordinatlari uzerinden calistigi icin hesaplanan acilar kameraya izdusen acilardir. Kamera acisi, perspektif, goruntu duzlemine dogru/duzlemden disari hareket ve keypoint tahmin hatalari aci degerlerini etkileyebilir. Bu nedenle degerler biyomekanik yorum icin yararlidir ancak tam 3D eklem rotasyonu olarak raporlanmamalidir.
