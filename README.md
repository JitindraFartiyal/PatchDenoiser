# PatchDenoiser
PatchDenoiser: Parameter-efficient multi-scale patch learning and fusion denoiser for medical images.

# Abstract
Low-dose CT images are essential for reducing radiation exposure in cancer screening, pediatric imaging, and longitudinal monitoring protocols where multiple scans are required over time. These images are critical for early lung cancer detection in high-risk patients, monitoring tumor response to therapy, evaluating treatment complications, and follow-up imaging in trauma and vascular disease. However, image quality is often degraded by noise arising from low-dose acquisition protocols, patient motion, or inherent scanner limitations, which can compromise diagnostic accuracy and negatively affect downstream image analysis. Conventional denoising approaches, such as filtering-based methods, frequently lead to excessive smoothing and loss of fine anatomical details. More recent deep learning-based techniques, including convolutional neural networks, generative adversarial networks, and transformer-based models, may help to overcome some of these limitations, but often struggle to preserve fine anatomical details or require large-scale models with substantial computational and energy costs,  limiting their practicality in clinical settings.

To address these challenges, we propose PatchDenoiser, a lightweight and energy-efficient multi-scale, patch-based denoising framework. PatchDenoiser decomposes the denoising task into local texture extraction and global context aggregation, which are subsequently fused through a spatially aware patch fusion strategy. This design enables effective noise suppression while preserving fine structural and anatomical details. The proposed model is designed to be ultra-lightweight, achieving significantly fewer parameters and substantially lower computational complexity compared to existing CNN, GAN, and transformer based denoising approaches.

Extensive experiments on the 2016 Mayo Low-Dose CT dataset demonstrate that PatchDenoiser consistently outperforms state-of-the-art CNN and GAN based methods in terms of PSNR and SSIM. The proposed framework also exhibits strong robustness across variations in slice thickness, reconstruction kernels, and Hounsfield unit (HU) windows, and generalizes effectively to cross-scanner data without requiring fine-tuning. Moreover, PatchDenoiser achieves approximately 9× fewer parameters and 27× lower energy consumption per inference compared with conventional CNN-based denoisers, underscoring its practical suitability for deployment in clinical AI pipelines.

Overall, PatchDenoiser offers a balanced combination of denoising performance, robustness, and computational efficiency, making it a practical and scalable solution for LDCT denoising in clinical settings.


# Architecture Design
## PatchDenoiser Main Diagram
![PatchDenoiser Overall Diagram](figs\PatchDenoiser Overall Diagram.png)

## Patch Fusion Module (PFM)
![Patch Fusion Module](figs\PFM.png)

# Results
![](figs\quantitative_results.png)