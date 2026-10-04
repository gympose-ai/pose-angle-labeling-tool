# easy_ViTPose — İnteraktif Poz Tahmini ve Düzenleyici

<p align="center">
  <img src="https://user-images.githubusercontent.com/24314647/236082274-b25a70c8-9267-4375-97b0-eddf60a7dfc6.png" width="340"/>
</p>

<p align="center">
  <b>İnsan Poz Tahmini · Eklem Noktası Düzenleme · İskelet Görselleştirme</b><br/>
  <a href="https://github.com/JunkyByte/easy_ViTPose">easy_ViTPose</a> üzerine inşa edilmiştir · Gradio 5 web arayüzü
</p>

<p align="center">
  <br/>
  <i>🏅 TÜBİTAK 1001 Projesi kapsamında geliştirilmiştir</i><br/>
  <b>Artistik Cimnastik Sporu için Poz Tabanlı Hareket Tespit Sistemi<br/>Tasarımı ve Prototip Gerçeklenmesi</b>
</p>

---

## Bu Proje Ne Yapıyor?

Bu proje, [easy_ViTPose](https://github.com/JunkyByte/easy_ViTPose) kütüphanesini **tam kapsamlı bir Gradio web arayüzü** ile genişletmektedir. Arayüz aracılığıyla şunları yapabilirsiniz:

1. Tek bir görüntü veya klasördeki tüm görüntüler üzerinde **ViTPose çıkarımı (inference)** çalıştırmak (ViTPose-H, COCO-25 modeli).
2. Tespit edilen eklem noktalarını canvas üzerinde **sürükleyerek interaktif biçimde düzenlemek**.
3. Omuz, dirsek, kalça, diz, ayak bileği ve ayak yönü için **eklem açılarını** görüntü üzerinde göstermek.
4. Düzenlenen sonuçları JSON dosyasına **kaydedip** aşağı akış boru hatlarına aktarmak.
5. Batch sonuçlarında araç çubuğuna gömülü **Prev / Next** kontrolleriyle kare kare gezinmek.
6. Mevcut bir JSON dosyasındaki poz koordinatlarını kullanarak **iskelet görselleştirmesi** yapmak (yeniden çıkarım gerekmez).

Bu araç; jimnastik, atletizm gibi alanlarda otomatik poz tahminlerinin manuel olarak düzeltilmesinin gerektiği **biyomekanik / spor bilimi araştırmaları** için tasarlanmıştır.

---

## Arayüze Genel Bakış

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Girdi Paneli                 │  İnteraktif Poz Düzenleyici (canvas)     │
│  ──────────────────────────── │  ──────────────────────────────────────  │
│  • Tek görüntü yükle  VEYA   │  • Eklem noktasını sürükleyerek düzelt   │
│  • Klasör yolunu yaz          │  • Kaydırma → Zoom / Boş alan sürükle →  │
│  • [Run ViTPose] butonu       │    Pan                                   │
│  • JSON yolu çıktı kutusu     │  • İsimler / Açılar / Noktalar aç-kapat  │
│                               │  • Tam ekran modu                        │
│                               │  • PNG anlık görüntü kaydet              │
│                               │  • ◀ Prev  │  Next ▶  (batch gezinti)   │
├─────────────────────────────────────────────────────────────────────────┤
│  [✅ Apply & Save]  ←  düzenlenen noktaları JSON'a yazar                │
│  Batch galeri (küçük resim şeridi, klasör modu)                         │
│  Kare / Görüntü kaydırıcısı                                             │
├─────────────────────────────────────────────────────────────────────────┤
│  JSON Yükleme → Çizim  (JSON'dan poz görselleştirme bölümü)             │
│  • Kaynak görüntü + mevcut .json yükle → iskeleti koordinatlardan çiz   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Poz Düzenleyici Kontrolleri

| Kontrol | İşlev |
|---|---|
| **Eklem noktasını sürükle** | Seçili eklemi hareket ettirir; değişiklik anında yansır |
| **Kaydırma tekerleği** | İmleç merkezli zoom açar / kapatır |
| **Boş alanı sürükle** | Canvas'ı kaydırır (pan) |
| **Boş alana çift tıkla** | Zoom ve pan'ı sıfırlar |
| **↩ Reset** | Orijinal eklem konumlarını geri yükler |
| **👁 Names** | Eklem adı etiketlerini gösterir / gizler |
| **💾 Save PNG** | Güncel canvas görünümünü PNG olarak indirir |
| **📐 Açılar** | Eklem açısı yay göstergelerini açar / kapatır |
| **∠ Özel Açı** | Standart listede olmayan bir açı için 3 keypoint seçerek açı hesaplar |
| **⤢ Tam Ekran** | Tam ekran moduna geçer |
| **● Noktalar** | Tüm eklem noktaları ve iskeleti gösterir / gizler |
| **◀ Prev / Next ▶** | Önceki / sonraki batch görüntüsüne geçer |
| **✅ Apply & Save** | Düzenlenen noktaları JSON dosyasına kaydeder |

---

## Proje Yapısı

```
easy_ViTPose/
├── app.py                      # Gradio uygulama giriş noktası
├── pose_editor.py              # İnteraktif canvas düzenleyici (HTML/CSS/JS + Python yardımcıları)
├── ANGLE_DEFINITIONS.md        # Eklem açılarının keypoint/vektör tanımları ve literatür notları
├── inference.py                # Komut satırı çıkarım betiği (orijinal easy_ViTPose)
├── export.py                   # ONNX / TensorRT dışa aktarma
├── model_split.py              # İnce ayar için ön eğitimli checkpoint dönüştürücü
├── evaluation_on_coco.py       # COCO değerlendirme betiği
├── setup.py                    # Paket kurulum dosyası
├── requirements.txt            # Python bağımlılıkları (CPU / genel)
├── requirements_gpu.txt        # GPU'ya özgü ek bağımlılıklar
├── requirements.notorch.txt    # PyTorch olmadan bağımlılıklar (özel kurulumlar)
├── Dockerfile                  # NVIDIA PyTorch container yapısı
├── checkpoints/                # ← model ağırlıklarını buraya koyun (git-ignored)
│   ├── vitpose-h-coco_25.pth
│   └── yolo11x.pt
├── easy_ViTPose/               # Temel kütüphane paketi
├── temp/                       # Çalışma zamanı geçici dosyalar (git-ignored)
└── outputs/                    # Çıkarım çıktıları (git-ignored)
```

---

## Ön Koşullar

### 1 — Python

**Python 3.9 – 3.11** önerilir. Python 3.12 bazı bilimsel paketlerle uyumluluk sorunları yaşayabilir.

### 2 — PyTorch

Geri kalan bağımlılıkları kurmadan önce **PyTorch ≥ 2.0**'ı manuel olarak kurun.  
Donanımınıza uygun sürümü seçin:

```bash
# CUDA 12.1 (NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Yalnızca CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Apple Silicon (MPS)
pip install torch torchvision torchaudio
```

> [!IMPORTANT]
> PyTorch, `pip install -r requirements.txt` komutundan **önce** kurulmalıdır.  
> Gereksinim dosyaları, doğru CUDA sürümünü seçebilmeniz için PyTorch'u kasıtlı olarak içermez.

### 3 — easy_ViTPose Kurulumu (Harici Bağımlılık)

Bu proje, çıkarım motoru olarak [easy_ViTPose](https://github.com/JunkyByte/easy_ViTPose) kütüphanesine bağımlıdır.  
Kütüphaneyi kaynaktan **düzenlenebilir paket** olarak kurmanız gerekmektedir:

```bash
# Upstream kütüphaneyi klonlayın
git clone https://github.com/JunkyByte/easy_ViTPose.git
cd easy_ViTPose

# Düzenlenebilir paket olarak kurun
pip install -e .
```

> [!NOTE]
> Doğrudan klonlanan `easy_ViTPose` deposu içinde çalışıyorsanız (yani bu README repo kökündeyse),  
> bu dizinden `pip install -e .` komutu yeterlidir — `easy_ViTPose` zaten kurulan pakettir.

---

## Kurulum

```bash
# 1. Bu repoyu klonlayın
git clone https://github.com/NecmettinTabak/Tubitak_1001.git
cd Tubitak_1001/easy_ViTPose

# 2. Önce PyTorch'u kurun (bkz. Ön Koşullar → adım 2)

# 3. Paketi düzenlenebilir modda kurun
pip install -e .

# 4. Geri kalan Python bağımlılıklarını kurun
pip install -r requirements.txt

# 5. (İsteğe bağlı) GPU'ya özgü ekstralar (ör. onnxruntime-gpu)
pip install -r requirements_gpu.txt

# 6. Gradio'yu kurun (henüz yüklü değilse)
pip install gradio>=5.0
```

### Temel Python Paketleri

| Paket | Amaç |
|---|---|
| `gradio >= 5.0` | Web arayüzü çerçevesi |
| `torch >= 2.0` | Derin öğrenme çıkarımı |
| `ultralytics` | YOLOv11 insan algılayıcı |
| `opencv-python` | Görüntü G/Ç ve çizim işlemleri |
| `scipy` | TPS warp (RBF interpolasyonu) |
| `numpy`, `Pillow` | Dizi ve görüntü yardımcıları |
| `onnxruntime` | İsteğe bağlı ONNX çıkarımı |

---

## Model Ağırlıklarını İndirme

Model checkpoint'lerini `checkpoints/` dizinine koyun (yoksa oluşturun):

```
checkpoints/
├── vitpose-h-coco_25.pth   # COCO + ayak veri setiyle eğitilmiş ViTPose-H (25 eklem noktası)
└── yolo11x.pt              # YOLOv11x insan algılayıcı
```

**Hugging Face** üzerinden indirin:  
👉 [https://huggingface.co/JunkyByte/easy_ViTPose](https://huggingface.co/JunkyByte/easy_ViTPose)

YOLO modeli için Ultralytics ilk çalıştırmada otomatik olarak indirir; ya da manuel indirebilirsiniz:

```bash
python -c "from ultralytics import YOLO; YOLO('yolo11x.pt')"
# Ardından yolo11x.pt dosyasını checkpoints/ dizinine taşıyın
```

---

## Uygulamayı Çalıştırma

```bash
python app.py
```

Gradio sunucusu yerel olarak başlar. Tarayıcınızda şu adresi açın:

```
http://127.0.0.1:7860
```

### Tek Görüntü Modu

1. **Input Image** alanına `.jpg` / `.png` / `.webp` dosyası yükleyin.
2. **Run ViTPose** düğmesine tıklayın.
3. Tespit edilen iskelet interaktif canvas üzerinde görünür.
4. Pozu düzeltmek için eklem noktalarını sürükleyin.
5. Düzeltilen koordinatları JSON'a yazmak için **✅ Apply & Save** düğmesine tıklayın.

### Batch / Klasör Modu

1. *Image Folder Path* alanına görüntüler içeren klasörün **tam yolunu** yazın.
2. **Run ViTPose** düğmesine tıklayın.
3. Tüm görüntüler işlenir; sonuçlar galeride belirir.
4. **◀ Prev / Next ▶** veya **Frame / Image** kaydırıcısıyla gezinin.
5. Her kareyi bağımsız olarak düzenleyin ve kaydedin.

### JSON Dosyasından Poz Çizimi (Alt Bölüm)

Bu bölüm, görüntü üzerinde modeli tekrar çalıştırmadan daha önce üretilmiş bir poz `.json` dosyasındaki koordinatları kullanır. Yani sistem yeni bir ViTPose çıkarımı yapmaz; JSON içinde kayıtlı eklem noktalarını okur, bu noktaları kaynak görüntünün üzerine yerleştirir ve iskelet çizimini bu verilerden oluşturur.

Bu kullanım özellikle daha önce çıkarımı yapılmış veya manuel olarak düzeltilip kaydedilmiş poz verilerini kontrol etmek için yararlıdır. Örneğin bir görüntüye ait JSON dosyasını yükleyerek koordinatların doğru kişiye, doğru eklemlere ve doğru görüntü boyutuna karşılık gelip gelmediğini görsel olarak inceleyebilirsiniz.

1. Pozu görselleştirilecek kaynak görüntüyü yükleyin.
2. Aynı görüntüye ait veya aynı koordinat sistemiyle hazırlanmış poz `.json` dosyasını yükleyin (easy_ViTPose çıktı formatında).
3. **Draw From Uploaded JSON** düğmesine tıklayın.
4. JSON dosyasındaki eklem koordinatları okunur ve iskelet görüntü üzerine çizilir.

---

## Çıktı JSON Formatı

Çıkarım çıktısı easy_ViTPose standart formatını izler:

```json
{
  "keypoints": [
    {
      "0": [
        [121.19, 458.15, 0.99],
        [110.02, 469.43, 0.98],
        "..."
      ]
    }
  ],
  "skeleton": {
    "0": "nose",
    "1": "left_eye",
    "2": "right_eye",
    "3": "left_ear",
    "4": "right_ear",
    "5": "neck",
    "6": "right_shoulder",
    "7": "left_shoulder",
    "8": "right_elbow",
    "9": "left_elbow",
    "10": "right_wrist",
    "11": "left_wrist",
    "12": "right_hip",
    "13": "left_hip",
    "14": "right_knee",
    "15": "left_knee",
    "16": "right_ankle",
    "17": "left_ankle",
    "18": "right_big_toe",
    "19": "right_small_toe",
    "20": "right_heel",
    "21": "left_big_toe",
    "22": "left_small_toe",
    "23": "left_heel",
    "24": "head_top"
  }
}
```

Her eklem noktası değeri görüntü piksel koordinatlarında `[y, x, güven_skoru]` biçimindedir.

`Apply & Save` sırasında ekranda görünen açılar JSON'a yazılır. Standart **Açılar** görünümü açıksa kırmızı standart açılar `standard_angles` alanına, manuel seçilen özel açılar `manual_angles` alanına kaydedilir. Tüm kayıtlar birlikte `angles` alanında da tutulur.

Gövde sapması, kol/bacak sapması, kolların geriye gidişi, kalça fleksiyon/ekstansiyon, diz ekstansiyon, iki bacak arası açı ve pelvis açısı değerleri ayrıca `derived_metrics` alanına kaydedilir. PDF/uzman mesajlarında istenen 18 teknik açı adı arayüzde ayrı butonlar olarak görünür; mevcut kısa teknik etiketler silinmeden korunur. Sağ/sol tarafı olan genel teknik açılar iki taraf hesaplanıp ortalanarak üretilir, aynı değerin farklı isimle kopyalanması yapılmaz.

Kalabalık görünümü azaltmak için **Açı Seç** düğmesi kullanılabilir. Bu modda listeden seçilen standart veya teknik metrikler görüntü üzerinde tek tek gösterilir; `Apply & Save` seçili mod açıkken yalnızca seçilen standart/teknik metrikleri JSON'a yazar. Eski **Açılar** düğmesi tüm açıları göstermeye ve kaydetmeye devam eder.

Her kayıt açının hangi üç keypoint ile hesaplandığını, merkez eklemi, vektörleri ve derece değerini içerir:

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

Aynı açı kayıtları ayrıca ayrı dosya olarak şu klasöre yazılır:

```text
easy_ViTPose/temp/açılar/
```

Dosya adı kaynak JSON'a göre üretilir ve `_angles.json` ile biter.

---

## Komut Satırı Çıkarımı (Gradio Olmadan)

Orijinal `inference.py` betiğini kullanarak komut satırından da çıkarım yapabilirsiniz:

```bash
python inference.py \
  --input ./test.png \
  --model ./checkpoints/vitpose-h-coco_25.pth \
  --yolo  ./checkpoints/yolo11x.pt \
  --dataset coco_25 \
  --model-name h \
  --save-img \
  --save-json \
  --output-path ./outputs
```

Tüm seçenekler için `python inference.py --help` komutunu çalıştırın.

---

## Docker

Container'ı derleyin (GPU desteği için NVIDIA Container Toolkit gereklidir):

```bash
docker build . -t easy_vitpose
```

Container içinde çıkarım çalıştırın:

```bash
docker run --gpus all --rm -it \
  --ipc=host \
  -v ./checkpoints:/checkpoints \
  -v ./inputs:/inputs \
  -v ./outputs:/outputs \
  easy_vitpose \
  python inference.py \
    --input /inputs/image.jpg \
    --model /checkpoints/vitpose-h-coco_25.pth \
    --yolo  /checkpoints/yolo11x.pt \
    --dataset coco_25 --model-name h \
    --save-img --save-json \
    --output-path /outputs
```

---

## İnce Ayar (Fine-tuning)

Ayrıntılar için upstream [easy_ViTPose ince ayar kılavuzuna](https://github.com/JunkyByte/easy_ViTPose#finetuning) bakın.  
Bu repoda yer alan yardımcı betikler:

- **`model_split.py`** — Resmi ViTPose checkpoint'ini tek başlıklı formata dönüştürür.
- **`evaluation_on_coco.py`** — Bir modeli COCO val2017 seti üzerinde değerlendirir.
- **`export.py`** — `.pth` checkpoint'ini ONNX / TensorRT formatına aktarır.

---

## Sorun Giderme

| Sorun | Çözüm |
|---|---|
| `ModuleNotFoundError: easy_ViTPose` | Repo kökünde `pip install -e .` komutunu çalıştırın |
| Model bulunamadı | `checkpoints/vitpose-h-coco_25.pth` dosyasının var olduğundan emin olun |
| YOLO indirme başarısız | `yolo11x.pt` dosyasını manuel indirip `checkpoints/` dizinine koyun |
| `GRADIO_DISABLE_BROTLI` uyarısı | `app.py` içinde zaten ayarlanmış; güvenle görmezden gelebilirsiniz |
| MPS'te yanlış sınırlayıcı kutu | `ultralytics`'i ≥ 8.2.48 sürümüne yükseltin |
| Canvas yüklenmiyor | Gradio ≥ 5.0 kullandığınızdan emin olun; düzenleyici `gr.HTML` ile `html_template` kullanmaktadır |

---

## Kaynaklar ve Referanslar

- ViTPose makalesi: [Y. Xu ve ark., 2022](https://arxiv.org/abs/2204.12484)
- Upstream kütüphane: [JunkyByte/easy_ViTPose](https://github.com/JunkyByte/easy_ViTPose)
- İnsan algılayıcı: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- SORT takipçisi: [abewley/sort](https://github.com/abewley/sort)
- COCO + Ayak veri seti: [CMU Perceptual Computing Lab](https://cmu-perceptual-computing-lab.github.io/foot_keypoint_dataset/)

---

[README](README.md) | [README_ENGLISH](README_english.md)
