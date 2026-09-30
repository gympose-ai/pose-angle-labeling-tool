# Doğrulanmış .venv Kurulumu

Bu belge, uygulamanın çalışan Windows ortamından alınan sürümlerle temiz ve tekrar üretilebilir kurulum yapmak içindir.

## Doğrulanan ortam

| Bileşen | Sürüm |
|---|---|
| Python | `3.11.9` |
| pip | `26.2.1` |
| PyTorch | `2.5.1+cu121` |
| torchvision | `0.20.1+cu121` |
| torchaudio | `2.5.1+cu121` |
| CUDA build | `12.1` |
| Gradio | `6.15.1` |
| NumPy | `1.26.0` |
| OpenCV | `4.8.0.76` |
| Pillow | `11.3.0` |

> NVIDIA kurulumu için güncel ve CUDA 12.1 ile uyumlu ekran kartı sürücüsü gerekir. PyTorch wheel gerekli CUDA runtime bileşenlerini getirir; ayrıca CUDA Toolkit kurmak çoğu durumda gerekli değildir.

## 1. Ön koşullar

- 64-bit Python `3.11.9`
- Git
- NVIDIA GPU kullanılacaksa güncel NVIDIA sürücüsü
- Video uyumluluğu için sistem PATH değişkeninde FFmpeg
- Aşağıdaki model dosyaları

```text
checkpoints/
├── vitpose-h-coco_25.pth
└── yolo11x.pt
```

Doğrulanan model boyutları:

| Dosya | Boyut |
|---|---:|
| `vitpose-h-coco_25.pth` | 2,549,009,242 bayt |
| `yolo11x.pt` | 114,636,239 bayt |

ViTPose modelleri: <https://huggingface.co/JunkyByte/easy_ViTPose>

## 2. Temiz sanal ortam oluşturma

PowerShell'i proje kökünde açın:

```powershell
py -3.11 --version
py -3.11 -m venv .venv
```

Ortamı etkinleştirin:

```powershell
.\.venv\Scripts\Activate.ps1
```

PowerShell script çalıştırmayı engellerse yalnızca mevcut terminal için:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Doğru ortamı kontrol edin:

```powershell
python --version
python -c "import sys; print(sys.executable)"
```

İkinci komut proje içindeki `.venv\Scripts\python.exe` yolunu göstermelidir.

## 3. pip sürümünü sabitleme

```powershell
python -m pip install --upgrade pip==26.2.1
```

## 4. PyTorch kurulumu

### NVIDIA GPU — doğrulanan kurulum

```powershell
python -m pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### Yalnızca CPU

```powershell
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

CPU kurulumu çalışır ancak özellikle video işleme belirgin şekilde daha yavaş olur.

## 5. Tam bağımlılık setini kurma

Çalışan `.venv` ortamının PyTorch dışındaki tüm paketleri tam sürümleriyle [requirements-venv-lock.txt](requirements-venv-lock.txt) dosyasına sabitlenmiştir:

```powershell
python -m pip install -r requirements-venv-lock.txt
```

Ardından bu projeyi mevcut kaynak koduna bağlı, düzenlenebilir paket olarak kurun:

```powershell
python -m pip install -e .
```

Bu adım önemlidir; uygulamadaki `easy_ViTPose` importları doğrudan bu çalışma klasörünü kullanır.

## 6. Kurulumu doğrulama

```powershell
python -m pip check
```

Beklenen çıktı:

```text
No broken requirements found.
```

Temel paketleri ve CUDA erişimini kontrol edin:

```powershell
python -c "import torch, gradio, cv2, numpy, PIL; print('torch:', torch.__version__); print('torch CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('gradio:', gradio.__version__); print('opencv:', cv2.__version__); print('numpy:', numpy.__version__); print('pillow:', PIL.__version__)"
```

NVIDIA kurulumu için beklenen değerler:

```text
torch: 2.5.1+cu121
torch CUDA build: 12.1
CUDA available: True
gradio: 6.15.1
opencv: 4.8.0
numpy: 1.26.0
pillow: 11.3.0
```

Model dosyalarını kontrol edin:

```powershell
Test-Path .\checkpoints\vitpose-h-coco_25.pth
Test-Path .\checkpoints\yolo11x.pt
```

Her iki komut da `True` dönmelidir.

## 7. Uygulamayı çalıştırma

```powershell
python app.py
```

Tarayıcı adresi:

```text
http://127.0.0.1:7860
```

## 8. Tek seferde kurulum özeti

NVIDIA GPU kullanılan doğrulanmış sıra:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip==26.2.1
python -m pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-venv-lock.txt
python -m pip install -e .
python -m pip check
python app.py
```

## 9. Sık karşılaşılan sorunlar

### `ModuleNotFoundError: easy_ViTPose`

```powershell
python -m pip install -e .
```

### `CUDA available: False`

```powershell
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

PyTorch sürümünde `+cu121` yoksa CPU paketi kurulmuş olabilir. Dördüncü bölümdeki NVIDIA komutunu tekrar çalıştırın.

### Video açılmıyor veya çıktı oynatılamıyor

```powershell
ffmpeg -version
python -c "import cv2; print(cv2.__version__)"
```

Kaynak videoyu mümkünse H.264 MP4 veya VP8/VP9 WebM biçiminde kullanın.

### Gradio arayüzü açılmıyor

```powershell
Get-NetTCPConnection -LocalPort 7860 -ErrorAction SilentlyContinue
```

Port başka bir süreç tarafından kullanılıyorsa o süreci kapatın veya uygulamaya farklı bir port verin.

## 10. Çalışan .venv paket listesi

Aşağıdaki liste `python -m pip freeze` çıktısından alınmıştır. Donanıma bağlı PyTorch kurulumu ayrı tutulduğu için gerçek kurulumda önce 4. bölümdeki PyTorch komutunu, ardından `requirements-venv-lock.txt` dosyasını kullanın.

<details>
<summary>Tam paket listesini göster</summary>

```text
aiofiles==24.1.0
annotated-doc==0.0.5
annotated-types==0.8.0
anyio==4.14.2
brotli==1.2.0
certifi==2023.7.22
charset-normalizer==3.2.0
click==8.5.0
colorama==0.4.6
coloredlogs==15.0.1
contourpy==1.1.1
cycler==0.11.0
-e git+https://github.com/gympose-ai/pose-angle-labeling-tool.git@fcea914dd1bec9b9d065bb76ad2e3ae1a9993a88#egg=easy_ViTPose
fastapi==0.128.1
ffmpeg==1.4
ffmpy==1.0.0
filelock==3.12.4
filterpy==1.4.5
flatbuffers==23.5.26
fonttools==4.43.0
fsspec==2026.7.0
gradio==6.15.1
gradio_client==2.5.0
groovy==0.1.2
h11==0.16.0
hf-gradio==0.4.1
hf-xet==1.6.0
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==1.33.0
humanfriendly==10.0
idna==3.4
imageio==2.31.3
importlib-resources==6.1.0
Jinja2==3.1.6
kiwisolver==1.4.5
lazy_loader==0.3
markdown-it-py==4.2.0
MarkupSafe==2.1.3
matplotlib==3.8.0
mdurl==0.1.2
mpmath==1.3.0
networkx==3.1
numpy==1.26.0
onnx==1.14.1
onnxruntime==1.16.0
opencv-python==4.8.0.76
orjson==3.12.0
packaging==26.3
pandas==2.1.1
pillow==11.3.0
protobuf==4.24.3
psutil==5.9.5
py-cpuinfo==9.0.0
pycocotools==2.0.8
pydantic==2.9.2
pydantic_core==2.23.4
pydub==0.25.1
Pygments==2.21.0
pyparsing==3.1.1
pyreadline3==3.5.6
python-dateutil==2.8.2
python-multipart==0.0.32
pytz==2023.3.post1
PyWavelets==1.4.1
PyYAML==6.0.1
requests==2.31.0
rich==15.0.0
ruff==0.16.9
safehttpx==0.1.7
scikit-image==0.21.0
scipy==1.11.2
seaborn==0.12.2
semantic-version==2.10.0
shellingham==1.5.4
six==1.16.0
starlette==0.47.0
sympy==1.13.1
tifffile==2023.9.18
tomlkit==0.13.3
torch==2.5.1+cu121
torchaudio==2.5.1+cu121
torchvision==0.20.1+cu121
tqdm==4.66.1
typer==0.27.2
typing_extensions==4.8.0
tzdata==2023.3
ultralytics==8.3.107
ultralytics-thop==2.2.0
urllib3==2.8.0
uvicorn==0.54.0
websockets==15.0.1
zipp==3.17.0
```

</details>
