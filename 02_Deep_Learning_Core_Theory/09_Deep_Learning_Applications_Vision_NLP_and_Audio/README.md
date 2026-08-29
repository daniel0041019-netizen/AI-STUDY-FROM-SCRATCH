# [Hands-on CV & Audio] Advanced Neural Mechanics & Signal Topology Optimization

Deep Learning 아키텍처와 디지털 신호 처리(DSP) 도메인에서 관례적으로 사용되던 기성 구현 방식의 한계를 깊이 있게 파고들고, 이를 **비유클리드 기하학, 스펙트럼 변환, 2D 위상 공간(Phase Space), C0 불연속성 검증** 등 새로운 연구적 시각으로 재해석하여 실증한 연구 중심의 트러블슈팅 리포트입니다.

Computer Vision의 **2D/3D Anchor Offset Regression** 및 **U-Net Feature Fusion**, Audio Signal Processing의 **Acoustic Trajectory Analysis** 및 **STFT Boundary Leakage**에 이르기까지, 모델 및 알고리즘 내부에 감춰져 있던 매개변수 간의 복잡한 수학적 상호작용과 병목 현상을 파헤치고 정량적 성능 지표로 증명한 핵심 모듈들을 담고 있습니다.

---

## 핵심 실험 및 트러블슈팅 요약

상세한 이론적 배경과 분석 리포트는 📄 **기술 블로그 포스팅/벨로그** 에 별도 정리되어 있습니다.

### 1. Polar Coordinate Anchor Offset Regression: 직교좌표계 분해를 통한 이동/변형 최적화 경로 독립
기존 RPN의 직교좌표계($(dx, dy, dw, dh)$) Bounding Box Regression은 객체의 중심점 이동(Translation)과 스케일/형태 변형(Scaling/Aspect Ratio)의 그래디언트가 오프셋 채널 간에 복잡하게 얽혀 학습 경로가 비선형적으로 왜곡됩니다. 본 리포트에서는 중심점 기준의 **극좌표계(Distance $r$, Angle $\theta$)와 등방성 스케일/종횡비 공간**으로 기하학적 매개변수를 완전 분리하여, 객체의 위치 이동과 형태 변형을 독립된 부분공간(Subspace)에서 최적화하는 아키텍처를 실증했습니다.

### 2. High-Frequency FFT Filtered Skip Fusion: 2D 스펙트럼 게이팅을 통한 세만틱 노이즈 차단
U-Net의 단순 Concatenation 스킵 연결은 수축 경로(Encoder)의 저주파 배경 노이즈(Context Noise)까지 디코더로 무분별하게 주입하여 채널 간 혼선을 유발합니다. 본 리포트에서는 **2D Real FFT를 기반으로 주파수 도메인 상에서 저주파 배경 성분을 차단(High-Pass Spectral Mask)하고, 오직 정교한 경계(High-Frequency Edge) 신호만 필터링하여 디코더로 전송하는 게이팅 스킵 메커니즘**을 설계하여 경계 신호 대 잡음비(SNR)를 극적으로 향상시켰습니다.

### 3. 2D Phase-Space Trajectory Dynamics: 스칼라 ZCR/STE 한계 극복을 위한 위상 공간 사영
스칼라 값인 Zero Crossing Rate(ZCR)와 Short-Time Energy(STE)는 1차원 도메인의 단순 통계 수치라 고주파 노이즈 환경에서 음향 경계가 모호해지는 치명적 한계가 있습니다. 이를 신호 $x(t)$와 1차 시간 미분(속도) $\frac{dx(t)}{dt}$로 구성된 **2D 위상 공간(Phase Space)**으로 확장하여, 유성음의 **주기적 타원 궤적(Orbital Energy)**과 무성음의 **불규칙 각속도 무질서도(Trajectory Chaoticity)**라는 위상 기하학적 차이로 음향 특성을 명확히 수치화했습니다.

### 4. STFT Boundary $C^0$-Discontinuity & Spectral Leakage Quantifier: 경계 절벽에 의한 고주파 산란 정량화
STFT 분석 시 윈도우 패딩 방식(`reflect` vs `constant`)에 따른 품질 차이는 단순 문헌적 경험칙이 아닌 **경계면에서의 0차 연속성($C^0$-discontinuity Step Jump Delta) 파괴**에 기인합니다. `constant`(0-padding)가 형성하는 경계면의 급격한 수직 절벽이 푸리에 변환 연산 시 고주파 대역으로 신호 에너지를 산산이 터뜨리는 스펙트럼 누설(Spectral Leakage)의 근본 원인임을 정량적 수치 지표로 엄밀히 증명했습니다.

---

## 핵심 실증 실험 (Experiments & Results)

### 실험 1. Polar Coordinate Anchor Offset Regression
* **파일명:** `01_spatial_scale_guided_anchor.ipynb`
* **실험 목적 (Why):** 직교좌표계 기반 앵커 오프셋 회귀 시 발생하는 중심 이동 연산과 스케일 변형 연산 간의 복잡한 그래디언트 간섭 현상을 완벽히 차단하고, 극좌표계 변환을 통해 좌표 복원 손실이 없는 기하학적 독립성을 실증합니다.
* **실험 방법 (How):** [40x40 Square] 기본 앵커 박스를 타겟 박스([20, 25, 80, 65])로 이동시키는 과정을 극좌표 변위 벡터($(dr, d_\theta)$) 및 등방성 면적/종횡비 오프셋($(d_{scale}, d_{aspect})$)으로 분해하여 역변환 디코딩 정밀도를 측정합니다.
* **실험 결과 (Result):** 위치 이동과 형태 변형 오프셋을 완전 분리한 상태에서도 Ground Truth 위치로의 디코딩 오차 없이 100% 가역성이 유지됨을 증명하여 오프셋 채널 간 디커플링을 성공적으로 달성했습니다.

### 실험 2. High-Frequency FFT Filtered Skip Fusion
* **파일명:** `02_high_frequency_gated_skip_fusion.ipynb`
* **실험 목적 (Why):** U-Net 스킵 연결 시 저주파 배경 노이즈가 주입되는 현상을 막기 위해, 2D FFT 기반 고주파 필터링을 도입했을 때 경계 신호 대 잡음비(SNR)의 비약적 향상 여부를 실측합니다.
* **실험 방법 (How):** 매끄러운 저주파 신호(Sin wave 배경)와 선명한 고주파 경계 신호(Vertical Line)가 섞인 합성 텐서 환경을 구축하고, 기존 단순 Concatenation과 본 제안 기법(HF-Gated Skip)을 거친 후 전송된 엣지 영역 Power 대비 배경 영역 Power 비율(SNR)을 정량 비교합니다.
* **실험 결과 (Result):** 표준 단순 결합 방식은 저주파 노이즈가 그대로 유입되어 낮은 SNR을 보인 반면, HF-Gated Skip 구조는 저주파 배경을 효과적으로 차단하여 **경계 신호 대 잡음비(SNR)를 +200% 이상 극적으로 향상**시킴을 정량 입증했습니다.

### 실험 3. 2D Phase-Space Trajectory Dynamics
* **파일명:** `03_phase_space_trajectory_dynamics.ipynb`
* **실험 목적 (Why):** 고주파 노이즈에 취약한 1차원 ZCR/STE 한계를 극복하기 위해, 오디오 프레임을 2D 위상 공간($(x(t), \frac{dx}{dt})$)으로 사영하여 유성음과 무성음의 동역학적 궤적 특성을 위상 지표로 분리합니다.
* **실험 방법 (How):** harmonic sinusoidal 신호(유성음)와 stochastic Gaussian white noise(무성음)에 대해 위상 공간 상의 반경 제곱 평균(Orbital Energy)과 궤적 각속도 변화량의 분산(Trajectory Chaoticity)을 실측 프로파일링합니다.
* **실험 결과 (Result):** 유성음은 위상 공간 내에서 10배 이상 높은 **Orbital Energy(0.3204)**를 유지하며 안정적인 주기 궤적을 형성한 반면, 무성음은 100배 이상 높은 **Trajectory Chaoticity(1.9121)**를 기록하여 노이즈 환경에서도 두 음향 성분이 명확히 분리됨을 증명했습니다.

### 실험 4. STFT Boundary Discontinuity & Leakage Quantifier
* **파일명:** `04_boundary_discontinuity_leakage_quantifier.ipynb`
* **실험 목적 (Why):** STFT 프레임 패딩 시 `constant`(0-padding) 방식이 유발하는 고주파 스펙트럼 누설(Spectral Leakage)의 근본 원인이 경계면에서의 불연속성(Step Jump Delta)임을 물리적으로 입증합니다.
* **실험 방법 (How):** 경계값이 0.8인 오프셋 신호에 대해 Constant 패딩과 Reflect 패딩을 각각 적용한 후, 경계 지점에서의 차분 절댓값($\vert{}x_{pad} - x_{raw}\vert{}$) 및 STFT 고주파 대역 밴드 에너지 스피어런스(High-Freq Leakage Power)를 정밀 측정합니다.
* **실험 결과 (Result):** Constant 패딩은 경계면에서 **0.8000의 치명적인 C0 Step Jump Delta**를 형성하며 0.0899의 고주파 누설 노이즈를 유발한 반면, Reflect 패딩은 연속적인 경계 재구성을 통해 **고주파 스펙트럼 누설을 18배 이상(0.0049) 획기적으로 감소**시켰음을 정량 입증했습니다.

---

## 핵심 데이터 수치 요약 (Analysis)

| 측정 및 분석 대상 | 데이터 스케일 / 환경 | 평가 및 실측 지표 | 성능 및 기하학적 민감도 진단 |
| :--- | :--- | :--- | :--- |
| **Polar Anchor Regression** | Anchor [40x40] $\rightarrow$ GT [60x40] | 좌표 복원 오차 및 오프셋 디커플링 | 오프셋 매개변수 완전 분리 상태에서 Reconstruction 오차 0.0 사수 |
| **HF-Gated Skip Fusion** | B=2, C=16, H=64, W=64 | Boundary SNR (Edge Power / BG Power) | Standard 대비 **경계 SNR 극적 향상 (+200% 이상)** 달성 |
| **2D Phase-Space Analysis** | Sample Rate 16kHz Audio Frame | Orbital Energy vs Trajectory Chaoticity | 유성음(High Orbital Energy)과 무성음(High Chaoticity) 간 궤적 완전 분리 |
| **STFT Boundary Leakage** | N_FFT=256, Window=Hann, Pad=128 | C0 Jump Delta & High-Freq Leakage | Reflect 패딩 적용 시 **고주파 스펙트럼 누설 18배 이상 감소** (0.0899 $\rightarrow$ 0.0049) |

---

## 기술 스택 및 라이브러리

* **Language:** Python 3.10+
* **Deep Learning & Spectral Acceleration:** PyTorch (`torch`, `torch.nn`, `torch.fft`, `torch.nn.functional`)
* **Numerical Computing & Signal Processing:** NumPy, SciPy
* **Analysis & Profiling Metrics:** Spectral Energy Density, Topological Trajectory Profiler

---

## 핵심 소스코드 아키텍처

### 1. High-Frequency FFT Filtered Skip Fusion Module (`02_high_frequency_gated_skip_fusion.ipynb`)

```python
import torch
import torch.nn as nn
import numpy as np

class HighFrequencyGatedSkip(nn.Module):
    """
    Applies a 2D FFT spectral high-pass mask to Encoder features before Skip Concatenation.
    Decouples structural boundary signals from low-frequency semantic noise.
    """
    def __init__(self, channels: int, cutoff_ratio: float = 0.25):
        super().__init__()
        self.cutoff_ratio = cutoff_ratio
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, encoder_feat: torch.Tensor, decoder_feat: torch.Tensor) -> torch.Tensor:
        # 1. Convert Spatial Encoder Features to Spectral Domain via 2D RFFT
        fft_feat = torch.fft.rfft2(encoder_feat, norm="backward")
        h, w = fft_feat.shape[-2:]
        
        # 2. Construct High-Pass Spectral Mask (Zero-out DC & Low Frequencies)
        mask = torch.ones((h, w), device=encoder_feat.device)
        cutoff_h, cutoff_w = int(h * self.cutoff_ratio), int(w * self.cutoff_ratio)
        mask[:cutoff_h, :cutoff_w] = 0.0  # Completely filter out low frequency background
        
        # 3. Inverse FFT to restore High-Frequency Edge Spatial Logits
        high_pass_fft = fft_feat * mask
        high_freq_spatial = torch.fft.irfft2(high_pass_fft, s=encoder_feat.shape[-2:], norm="backward")
        
        # 4. Context-Aware Gating from Decoder
        gate = self.spatial_gate(decoder_feat)
        filtered_encoder_feat = high_freq_spatial * gate
        
        # 5. Precise Edge Signal + Decoder Feature Concatenation
        return torch.cat([decoder_feat, filtered_encoder_feat], dim=1)


# =====================================================================
# Hypothesis Proof Metric: Boundary Signal-to-Noise Ratio (SNR) Test
# =====================================================================
if __name__ == "__main__":
    B, C, H, W = 2, 16, 64, 64

    # 1. Synthetic Feature Map: Smooth Low-Frequency Background + Sharp High-Frequency Edge
    grid_y, grid_x = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing='ij')
    low_freq_bg = torch.sin(2 * np.pi * grid_x) * torch.sin(2 * np.pi * grid_y)
    high_freq_edge = (torch.abs(grid_x - 0.5) < 0.02).float()

    synthetic_enc = (low_freq_bg + high_freq_edge).unsqueeze(0).unsqueeze(0).repeat(B, C, 1, 1)
    synthetic_dec = torch.ones_like(synthetic_enc)

    # 2. Standard Concat vs HF-Gated Skip Processing
    standard_passed = synthetic_enc
    hf_skip_module = HighFrequencyGatedSkip(channels=C)
    hf_skip_passed = hf_skip_module(synthetic_enc, synthetic_dec)[:, C:, :, :]

    # 3. Calculate Signal-to-Noise Ratio (SNR = Edge Power / Background Power)
    edge_mask = (high_freq_edge > 0.5).unsqueeze(0).unsqueeze(0).repeat(B, C, 1, 1)
    bg_mask = (high_freq_edge <= 0.5).unsqueeze(0).unsqueeze(0).repeat(B, C, 1, 1)

    std_edge_pwr = standard_passed[edge_mask].pow(2).mean().item()
    std_bg_pwr = standard_passed[bg_mask].pow(2).mean().item()
    std_snr = std_edge_pwr / (std_bg_pwr + 1e-8)

    hf_edge_pwr = hf_skip_passed[edge_mask].pow(2).mean().item()
    hf_bg_pwr = hf_skip_passed[bg_mask].pow(2).mean().item()
    hf_snr = hf_edge_pwr / (hf_bg_pwr + 1e-8)

    # 4. Verification Output
    print("=" * 60)
    print("[CONCLUSION] High-Frequency Gated Skip Verification Results")
    print("=" * 60)
    print(f"Standard Concat -> Edge Power: {std_edge_pwr:.4f} | BG Power: {std_bg_pwr:.4f} | SNR: {std_snr:.2f}")
    print(f"HF-Gated Skip   -> Edge Power: {hf_edge_pwr:.4f} | BG Power: {hf_bg_pwr:.4f} | SNR: {hf_snr:.2f}")
    print("-" * 60)
    print(f"-> Boundary Signal-to-Noise Ratio Improved by +{(hf_snr / std_snr - 1) * 100:.1f}%")
    print("=" * 60)
