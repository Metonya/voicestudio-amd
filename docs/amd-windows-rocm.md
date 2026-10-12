# CUDA kütüphanelerini Windows'ta AMD (ROCm) ile çalıştırma — Araştırma raporu

Hedef donanım: **AMD Radeon RX 7900 XT** (mimari: RDNA3, ROCm kodu: **`gfx1100`**, 20 GB VRAM), Windows.
Tarih: 2026-10-12.

---

## 1. Kısa cevap

- **CUDA kodu AMD kartta "olduğu gibi" çalışmaz.** CUDA, NVIDIA'ya özel. Ama 2026 itibarıyla **çoğu durumda çalıştırmak mümkün**.
- **En önemli gerçek:** VoiceStudio gibi Python projelerinin büyük kısmı CUDA kodu yazmaz, **PyTorch** kodu yazar. PyTorch'un **ROCm sürümü**, `torch.cuda` API'sini aynen taklit eder. Yani `device="cuda"` yazan kod, **hiç değişmeden** AMD kartta çalışır. Sadece doğru `torch` paketini kurmak gerekir.
- **2026'da büyük değişiklik:** AMD artık **Windows için resmi PyTorch paketleri** yayınlıyor. RX 7900 XT (`gfx1100`) Windows'ta **"Release Ready ✅"** durumunda. Eskiden bu iş için Linux/WSL veya ZLUDA şarttı; artık değil.
- **Sorun çıkaranlar:** PyTorch'un kendisi değil, onun yanındaki **hazır derlenmiş CUDA eklentileri** (CTranslate2, flash-attn, bitsandbytes, deepspeed, xformers, onnxruntime-gpu, cupy, TensorRT…). Her biri için ayrı ROCm sürümü olup olmadığına bakmak gerekiyor.
- **Sonuç: VoiceStudio için bu iş yapılabilir.** Repo Linux'ta ROCm'u zaten destekliyor; Windows'ta bilerek kapatılmış (aşağıda detay). Kapıyı açmak için gereken parçalar artık mevcut.

---

## 2. AMD kartta CUDA çalıştırmanın 4 yolu

| Yol | Nasıl çalışır | Windows'ta durum (Ekim 2026) | Ne zaman kullanılır |
|---|---|---|---|
| **1. Native ROCm/HIP** (önerilen) | Kütüphanenin AMD için derlenmiş sürümü kullanılır. PyTorch ROCm, `torch.cuda` gibi davranır. | **Resmi destek var.** AMD ROCm 7.2.1 (torch 2.9.1) ve TheRock/ROCm 7.14 (torch 2.11–2.14). `gfx1100` ✅ | PyTorch tabanlı her şey. **Ana yol bu.** |
| **2. ZLUDA** (çeviri katmanı) | Derlenmiş CUDA programını **değiştirmeden** AMD'de çalıştırır (CUDA çağrılarını HIP'e çevirir). | v6 (Haziran 2026), v7-preview sürüyor. llama.cpp tam çalışıyor. **PyTorch desteği hâlâ "sağlam" değil.** Proje **fonsuz kaldı, hobi projesi**. | Kaynak kodu olmayan, sadece CUDA binary'si olan programlar. **Son çare.** |
| **3. Kaynak çevirme** (HIPIFY / SCALE) | `.cu` kaynak kodunu HIP'e çevirip AMD için derler. | HIPIFY Windows'ta mevcut (TheRock içinde ✅). SCALE (Spectral Compute) Linux odaklı, Windows'u doğrulayamadım. | Kendi CUDA kernel'i olan kütüphaneler (ör. özel `.cu` dosyaları). |
| **4. CUDA'sız alternatif** | Vulkan (llama.cpp/ggml), ONNX Runtime + DirectML/Windows ML | Vulkan **iyi çalışıyor**. **DirectML bakım modunda** (yeni özellik yok), `torch-directml` eski torch'a kilitli. | Modelin GGUF/ONNX sürümü varsa. |

Ek seçenek: **WSL2 + Linux ROCm.** RX 7900 XT WSL2'de destekleniyor. VoiceStudio'nun `:rocm` Docker imajı bunu zaten destekliyor (`docs/install/docker.md` → "AMD GPU on WSL2").

### Neden ZLUDA ana yol değil?
- PyTorch gibi büyük kütüphanelerde hâlâ eksikler var; geliştirici 2025 sonu hedefini tutturamadığını kendisi yazdı.
- Proje fon kaybetti, tek kişilik hafta sonu projesi. Uzun vadede güvenilmez.
- Native ROCm artık Windows'ta var. Çeviri katmanına gerek kalmadı.

### Neden DirectML değil?
- Microsoft DirectML'i **bakım moduna** aldı. Yeni özellik gelmeyecek, yerine Windows ML öneriliyor.
- `torch-directml` eski torch sürümlerine kilitli. Benzer bir projede (Voicebox) kullanıcılar DirectML ile **yanlış sonuçlar** aldıklarını raporladı.

---

## 3. "CUDA projesini kolayca ROCm'a çevirebilir miyiz?" — Karar ağacı

Her kütüphane için şu soruları sırayla sor:

1. **Saf PyTorch mı?** (sadece `torch`, `torchaudio`, `transformers` vb.)
   → **Çevirmeye gerek yok.** ROCm torch kur, bitti. `torch.cuda.is_available()` → `True` döner.
2. **Hazır derlenmiş CUDA eklentisi var mı?** (ör. `ctranslate2`, `flash_attn`, `bitsandbytes`)
   → ROCm sürümü var mı bak (tablo aşağıda).
   → Yoksa ve **opsiyonelse kapat** (ör. flash-attn yerine PyTorch'un kendi `SDPA`'sı).
   → Opsiyonel değilse 3. adıma geç.
3. **Kendi `.cu` kaynak kodu var mı?**
   → ROCm torch ile `torch.utils.cpp_extension` kullanan eklentiler **derlenirken otomatik hipify edilir.** Çoğu zaman sadece yeniden derlemek yeter.
   → Değilse `hipify-perl` / `hipify-clang` ile çevir, elle düzelt.
   → İyi haber: RDNA3 varsayılan olarak **wave32** çalışır (NVIDIA'nın warp size'ı da 32). Warp'a bağlı kodlar çoğu zaman bozulmaz.
   → Zor kısımlar: inline **PTX** assembly, `cuDNN`/`TensorRT`/`cuSPARSELt` gibi özel kütüphaneler.
4. **Sadece binary var, kaynak yok mu?** → **ZLUDA** dene.
5. **Hiçbiri olmuyor mu?** → WSL2 + Linux ROCm, veya modelin Vulkan/ONNX sürümü.

### Kütüphane durum tablosu (Windows + `gfx1100`)

| Kütüphane | Windows ROCm durumu | Not |
|---|---|---|
| **torch / torchaudio / torchvision** | ✅ Resmi | AMD 7.2.1: torch 2.9.1, **sadece Python 3.12**. TheRock (ROCm 7.14+ / 10.x): torch 2.12–2.14, **Python 3.11–3.14**. |
| **SDPA / aotriton** (attention hızlandırma) | ✅ Destekli | flash-attn yerine bunu kullan. |
| **triton / `torch.compile`** | 🟠 Yeni / kısmen | AMD'nin triton-windows çalışması Temmuz 2026'da "tamamlandı" diye kapandı, ama TheRock tablosu hâlâ "in progress" diyor. **Test edilmeli.** |
| **CTranslate2** (faster-whisper, whisperx) | ✅ v4.7.0'dan beri resmi | Wheel'ler **PyPI'da değil**, GitHub release'te (`rocm-python-wheels-Windows.zip`). İçini açtım: **`gfx1100` kernel'i var**, `amdhip64_7.dll` + `hipblas.dll` istiyor. |
| **MIOpen** (conv katmanları) | ✅ | İlk çalışmada kernel araması yavaş olabilir (repo bunu zaten ayarlıyor, #2373). |
| **flash-attn** | 🟠 Deneysel | Triton backend ile topluluk rehberleri var. RDNA3'te backward (eğitim) eksik. |
| **bitsandbytes** | ❌ Windows'ta resmi yok | Linux'ta ROCm build var. Windows'ta topluluk denemeleri var. |
| **deepspeed, apex, FBGEMM GenAI** | ❌ | Windows'ta yok. Çıkarım (inference) için genelde gerekmez. |
| **onnxruntime-gpu** (CUDA EP) | ❌ | AMD için DirectML EP veya MIGraphX EP gerekir. |
| **amd-smi / rocm-smi** | ❌ Windows'ta yok | VRAM bilgisi için `torch.cuda.mem_get_info()` kullanılabilir. |
| **llama.cpp / ggml** | ✅ | Vulkan ve HIP backend'leri Windows'ta çalışıyor. |

---

## 4. VoiceStudio repo'sunun mevcut durumu

**Zaten var olanlar (Linux / WSL2 / Docker):**
- `backend/core/device_caps.py`: ROCm'u `torch.version.hip` ile tanıyor, `gfx` mimari kontrolü ve `HSA_OVERRIDE_GFX_VERSION` mantığı var.
- `backend/core/torch_indexes.py`, `scripts/setup.py`: `OMNIVOICE_TORCH_VARIANT=rocm` ile ROCm torch kurulumu (Linux).
- Docker `:rocm` imajı ve WSL2 `/dev/dxg` köprüsü.
- Vulkan kullanan motorlar: `audiocpp`, `omnivoice_gguf` (Windows'ta Radeon'u kullanabilir).

**Windows'ta neden kapalı?** `electron/src/main/runtime-project.ts` içindeki yorum:

> "PyTorch's ROCm index publishes Linux wheels only. (...) **Windows ROCm needs AMD's own wheels and a validated engine matrix**; it is deliberately not wired into this recipe."

Yani: `rocmTorchApplies()` Windows'ta `false` döner. **Bu fork'un yapacağı iş tam olarak bu cümledeki iki eksik:** AMD'nin Windows wheel'lerini bağlamak ve motorları tek tek test etmek.

**Önümüzdeki engeller:**

| Engel | Neden sorun | Çözüm önerisi |
|---|---|---|
| **torch sürümü** | Repo `torch==2.8.0`'a kilitli. AMD Windows'ta 2.9.1+ veriyor. | ROCm-Windows için ayrı pin seti (Linux'taki gibi ayrı varyant). |
| **Python sürümü** | Repo **Python 3.11** kullanıyor. AMD'nin eski 7.2.1 Windows wheel'leri sadece 3.12. | **Engel kalktı:** TheRock (ROCm 7.14+) Windows'ta Python 3.11–3.14 destekliyor. TheRock kullanılacak. |
| **CTranslate2 (ASR)** | Repo ROCm'da CTranslate2'yi CPU'ya yönlendiriyor (#1529, "HIP backend yok" varsayımı artık **eski**). | 4.8.2 ROCm wheel'ini dene. DLL'ler torch'un ROCm paketlerinden bulunabilir mi test et. Yedek: pytorch-whisper zaten GPU'da çalışıyor. |
| **iGPU tuzağı** | Ryzen 7000/9000 işlemcilerde dahili GPU (ör. `gfx1036`) `cuda:0` olarak görünüp **her şeyi bozabiliyor** (`hipErrorInvalidImage`, TheRock #5543). Düzeltme Eylül 2026'da geldi. | `HIP_VISIBLE_DEVICES` ile iGPU'yu gizle, veya yeni wheel kullan. Uygulama bunu otomatik yapmalı. |
| **Performans** | Benzer proje Voicebox'ta, Windows'ta "çıplak" ROCm, autoregressive TTS'te **CPU'dan hızlı değildi**. SDPA + overhead azaltma + HIP graph ile ~10× hızlandı. | SDPA'yı zorla, CUDA graph yolunu (`omnivoice_flashinfer`) HIP'te test et. Her motoru CPU'ya karşı **RTF** ile ölç. |
| **`torch.compile`** | Triton Windows+AMD yeni. | Çalışmazsa kapat. CLAUDE.md'ye göre bu bir parity ihlali değil (performans, davranış değil). |

---

## 5. Önerilen mimari

```
                 ┌─────────────────────────────────────────────┐
                 │  Electron setup (runtime-project.ts)        │
                 │  GPU tespiti → Radeon + Windows             │
                 └───────────────┬─────────────────────────────┘
                                 │  torch varyantı: "rocm-win"
                                 ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  Python 3.12 venv                                            │
   │  torch / torchaudio  ← AMD Windows wheel'leri                │
   │     (stabil: repo.radeon.com 7.2.1  |  yeni: TheRock 7.14+) │
   │  rocm-sdk-* paketleri (HIP runtime, hipBLAS, MIOpen DLL'leri)│
   │  ctranslate2 (ROCm wheel, opsiyonel)                         │
   └───────────────┬──────────────────────────────────────────────┘
                   │  torch.cuda.* = AMD GPU (HIP)
                   ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  backend: device_caps → family="rocm" (zaten var)            │
   │  engine_routing → sadece "doğrulanmış" motorlar GPU'ya       │
   │  diğerleri → CPU (dürüst açıklama ile)                       │
   └──────────────────────────────────────────────────────────────┘
   Yedek yollar:  Vulkan motorları (audiocpp, gguf)  |  WSL2 + Docker :rocm
```

**Prensipler:**
1. **Native ROCm ana yol.** ZLUDA ve DirectML ana yola girmez.
2. **Opt-in başla.** Önce `OMNIVOICE_TORCH_VARIANT=rocm` ile Windows'ta açılsın. Doğrulanınca otomatik tespit.
3. **Motor bazında doğrulama listesi.** Her motor "GPU'da test edildi" işareti alana kadar CPU'da kalır. Bu, repo'nun mevcut "dürüst raporlama" yaklaşımıyla uyumlu.
4. **Upstream'e uyumlu kal.** Değişiklikleri Linux ROCm yolunun yanına ekle, ayrı kod yolu açma. Upstream'e PR olarak gönderilebilir hale getir.

---

## 6. Deneme planı (senin bilgisayarında)

**Aşama 0 — Bilgi topla**
- Windows sürümü (10/11), Adrenalin sürümü, işlemci modeli (**dahili GPU var mı?**).

**Aşama 1 — Çıplak PyTorch testi** (VoiceStudio'dan bağımsız, temiz klasörde)
```powershell
py -3.12 -m venv rocmtest
rocmtest\Scripts\activate
pip install --index-url https://nightly.repo.amd.com/rocm/whl-next/ "torch[device-gfx1100]" "torchvision[device-gfx1100]" torchaudio
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Bakılacaklar: matmul, SDPA, conv1d (MIOpen), torchaudio, VRAM okuma.

**TheRock kanalları** (Python paket index'leri):

| Kanal | Index | Not |
|---|---|---|
| **Stable** (uygulama için) | `https://stable.repo.amd.com/rocm/whl-next/` | ROCm 10.0+ stabil |
| Nightly (deneme için) | `https://nightly.repo.amd.com/rocm/whl-next/` | Her gün değişir, eski sürümler silinebilir |
| Eski stable (7.14) | `https://repo.amd.com/rocm/whl-multi-arch/` | torch 2.12, Python 3.11–3.14 |

**Aşama 2 — VoiceStudio OmniVoice TTS**
- Aynı venv'e VoiceStudio bağımlılıklarını kur, OmniVoice'u GPU'da çalıştır.
- CPU ile karşılaştır: **RTF** (üretim süresi / ses süresi).

**Aşama 3 — ASR**
- CTranslate2 4.8.2 ROCm wheel + faster-whisper.
- pytorch-whisper yolu.

**Aşama 4 — Diğer motorlar**
- IndexTTS, CosyVoice, VoxCPM2, MOSS, demucs, pyannote, watermark.
- Her biri için: çalışıyor mu, hız, VRAM, hata.

**Aşama 5 — Uygulamaya entegrasyon**
- `runtime-project.ts`, `scripts/setup.py`, `torch_indexes.py`: Windows ROCm varyantı.
- iGPU gizleme, CTranslate2 ROCm yolu, motor doğrulama listesi, testler, dokümantasyon.

---

## 6.1 Test sonuçları (2026-10-12)

Sistem: Windows 11 25H2, RX 7900 XT, Adrenalin 26.8.1, Ryzen 9 7900X (dahili GPU var), 32 GB RAM.

| Test | Sonuç | Not |
|---|---|---|
| ROCm torch kurulumu (TheRock nightly) | ✅ | `2.14.0+rocm10.2.0a20261011`, HIP 7.17, `cuda:0` = RX 7900 XT |
| Matmul, SDPA, Conv1d (MIOpen), torchaudio Mel, VRAM okuma | ✅ | 19.98 GB VRAM görünüyor |
| fp16 matmul (8192²) | ✅ **63.8 TFLOPS** | hipBLASLt düzgün çalışıyor |
| SDPA | ✅ | FLASH (aotriton) 0.71 ms, EFFICIENT 0.75 ms, MATH 4.80 ms |
| Kernel launch gecikmesi | ✅ **2.6 µs/op** | CUDA ile aynı seviye. HIP graph çalışıyor ama kazanç yok (3.5 µs/op). |
| Dahili GPU | ⚠️ | `gfx1036` 1 numaralı cihaz olarak görünüyor. RX 7900 XT 0 numara olduğu için sorun çıkarmadı. |
| MIOpen uyarıları | ⚠️ Zararsız | `gfx110054` = gfx1100 + 84 CU (hex `54`) için hazır performans veritabanı yok. |
| PyTorch Whisper large-v3-turbo (GPU fp16) | ✅ **RTF 0.075** | 8.28 s ses: ilk çalıştırma 9.85 s, ısınmış 0.62 s |
| PyTorch Whisper large-v3-turbo (CPU fp32) | ❌ RTF 25.9 | 215 s. Normalin çok üstünde. Şüphe: aynı süreçte iki OpenMP runtime (`libomp140` + CTranslate2'nin `libiomp5md`). Probe güncellendi. |
| CTranslate2 4.8.2 ROCm + faster-whisper `tiny` (GPU) | ✅ **ısınmış 0.14 s** | Sadece torch önce import edilince. Önce edilmezse `ctranslate2.dll` bağımlılıkları bulunamıyor (CPU modunda bile). |

**Kök neden (CTranslate2):** ROCm DLL'leri (`amdhip64_7.dll`, `hipblas.dll`) torch'un pip paketlerinin içinde. `import torch` bu klasörleri DLL arama yoluna ekliyor. Torch'tan önce yüklenen CTranslate2 bunları bulamıyor. İlk testteki `hipErrorInvalidKernelFile` muhtemelen sürücünün `System32` içindeki `amdhip64_7.dll`'i ile wheel'in kütüphanelerinin karışmasıydı.

**Yapılan düzeltme:** `backend/core/rocm_windows.py` ROCm DLL klasörlerini CTranslate2'den önce kaydediyor. Ayrıca CTranslate2'nin HIP sürümü kuruluysa ROCm'da GPU kullanılıyor (`_ctranslate2_cuda_ok`, ASR sidecar, routing raporu), PyTorch Whisper ROCm'u destekliyor olarak işaretlendi.

**VoiceStudio venv'i (Python 3.11) ile kurulum:**
- Stable kanalda Python 3.11 paketleri var: `torch==2.14.0+rocm10.1.0`, `torchvision==0.29.0a0+rocm10.1.0`, `torchaudio==2.11.0.3+rocm10.1.0` (HIP 7.16).
- **Versiyon yazmadan kurmak işe yaramıyor:** `torch 2.8.0+cpu` zaten "kurulu" sayılıyor ve pip hiçbir şey yapmıyor. Versiyonu açıkça yazmak gerekiyor.
- CTranslate2 için `ctranslate2-4.8.2-cp311-cp311-win_amd64.whl` (GitHub release zip'inden) çalışıyor.
- Uygulama ilk denemede "Python environment is incomplete" dedi. Sebep: Electron'un açılış kontrolü `faster_whisper`'ı torch'tan önce import ediyordu. Kontrolün sırası backend ile aynı yapıldı (önce torch). Backend de açılışta ROCm DLL klasörlerini kaydediyor.

Açık konu: CPU yavaşlığı → yeni probe ile ölçülecek.

---

## 7. Kaynaklar

- TheRock (AMD'nin yeni ROCm build sistemi): [RELEASES.md](https://github.com/ROCm/TheRock/blob/main/RELEASES.md), [SUPPORTED_GPUS.md](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md), [Windows desteği](https://github.com/ROCm/TheRock/blob/main/docs/development/windows_support.md), [PyTorch özellik tablosu](https://github.com/ROCm/TheRock/blob/main/external-builds/pytorch/README.md)
- [TheRock #5543 — RX 7900 XT + iGPU hatası](https://github.com/ROCm/TheRock/issues/5543), [TheRock legacy release index'leri](https://github.com/ROCm/TheRock/blob/main/docs/packaging/legacy_multi_arch_releases.md), [ROCm 7.14 kurulum (Python 3.11–3.14)](https://rocm.docs.amd.com/en/docs-7.14.0/install/rocm.html), [TheRock #8531 — gfx1100 Windows access violation](https://github.com/ROCm/TheRock/issues/8531)
- [ROCm 7.14 duyurusu (TheRock production)](https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-7.14-blog/README.html), [Phoronix: ROCm 7.14](https://phoronix.com/news/AMD-ROCm-7.14)
- [AMD Windows uyumluluk tablosu](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html), [AMD Windows PyTorch kurulumu (7.2.1)](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/install/installryz/windows/install-pytorch.html), [AMD PyTorch on Windows 7.2 release notes](https://www.amd.com/en/resources/support-articles/release-notes/RN-AMDGPU-WINDOWS-PYTORCH-7-2.html), [AMD AI Bundle](https://www.amd.com/en/resources/support-articles/faqs/ai_bundle.html)
- ZLUDA: [releases](https://github.com/vosen/ZLUDA/releases), [Q1&Q2 2026 güncellemesi](https://vosen.github.io/ZLUDA/blog/zluda-update-q1q2-2026/), [Q4 2025 güncellemesi](https://vosen.github.io/ZLUDA/blog/zluda-update-q4-2025/), [Tom's Hardware: ZLUDA v6 fonsuz kaldı](https://www.tomshardware.com/pc-components/gpu-drivers/cuda-emulator-for-amd-gpus-zluda-loses-funding-with-v6-release-embattled-project-goes-back-to-hobby-status-but-now-includes-32-bit-physx-support), [ComfyUI-Zluda](https://github.com/patientx/ComfyUI-Zluda)
- CTranslate2: [releases](https://github.com/OpenNMT/CTranslate2/releases), [#2016 Windows DLL isim sorunu](https://github.com/OpenNMT/CTranslate2/issues/2016)
- [triton-windows #2 — AMD desteği](https://github.com/triton-lang/triton-windows/issues/2)
- [Voicebox #676 — benzer TTS uygulamasında Windows AMD deneyimi](https://github.com/jamiepine/voicebox/issues/676)
- [Microsoft DirectML (bakım modu)](https://github.com/microsoft/DirectML)
- [bitsandbytes kurulum dokümanı](https://huggingface.co/docs/bitsandbytes/v0.50.0/installation)
- [SCALE (Spectral Compute) — Phoronix](https://www.phoronix.com/news/SCALE-CUDA-Apps-For-AMD-GPUs)
- [Topluluk rehberi: ROCm 7 + triton + flash-attention (Windows, RDNA3)](https://github.com/guinmoon/rocm7-triton-flash-attention-sage-attention-bnb)
