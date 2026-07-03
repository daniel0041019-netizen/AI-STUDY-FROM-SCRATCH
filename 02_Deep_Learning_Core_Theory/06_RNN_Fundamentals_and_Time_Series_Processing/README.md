# [Deep Learning] 06. CNN Architecture & Deep Network Optimization

《핸즈온 머신러닝(14장)》 및 《딥러닝의 정석(5장)》의 깊게 탐독하는 과정에서 마주한 구조적 의문들을 정리하고, 심층 신경망과 합성곱 신경망(CNN)을 관통하는 핵심 메커니즘을 시각화하고 직접 실험 및 증명한 보고서입니다. 단순한 '기성 관례(Convention)'로 치부되던 레이어 배치법부터 지식 증류(Knowledge Distillation)의 수치적 안정성, CAM의 수학적 펀더멘탈까지 직접 파고들며 그 속에 숨겨진 최적화의 원리를 본질적으로 해체해 본 트러블슈팅 리포트입니다.

## 핵심 실험 및 트러블슈팅 요약

상세한 이론적 배경과 분석 리포트는 📄 [기술 블로그 포스팅/벨로그](https://velog.io/@daniel3721/Deep-Learning-02.-%EC%95%84%ED%82%A4%ED%85%8D%EC%B2%98%EC%9D%98-%EC%88%A8%EC%9D%80-%EC%88%98%ED%95%99%EC%A0%81-%EC%9D%98%EB%8F%84%EC%99%80-%EB%AA%A8%EB%8D%B8-%EC%95%95%EC%B6%95%ED%95%B4%EC%84%9D%EC%9D%98-%EB%B3%B8%EC%A7%88) 에 별도 정리되어 있습니다.

### 1. Batch Normalization과 Dropout의 기하학적 간섭: 왜 BN은 Conv-ReLU 사이에, Dropout은 블록 끝에 위치해야 하는가?
딥러닝 아키텍처를 설계할 때 흔히 배치 정규화(BN)와 드롭아웃(Dropout)을 무지성으로 혼용하곤 합니다. 그러나 이 두 규제 기법을 잘못 배치하면 성능이 처참하게 망가지는 '배치 정규화와 드롭아웃의 조화적 모순(Disharmony)'을 겪게 됩니다. 만약 Dropout이 BN 앞단에 위치하면 무작위로 뉴런을 꺼버릴 때마다 미니배치의 내부 통계량이 요동치게 되며, 테스트 시점에 전체 뉴런을 모두 켜고 이동평균 통계량을 적용하면 학습 시 계산했던 통계량과 극심한 불일치(Inference Shift)가 발생하여 모델의 예측력이 붕괴됩니다. 데이터 오염과 통계량 왜곡을 막기 위해 합성곱 블록 내부는 `Conv → BN → ReLU → Dropout (Block End)`의 엄격한 위계를 따라야 함을 실증했습니다.

### 2. 레이어 배치와 깊이의 임계점: 왜 VGG는 3×3을 '3개' 쌓았고, ResNet은 '34층/50층'에서 끊었을까?
VGG-16에서 초반 특징 추출 시 3×3 커널을 4개 이상 연속으로 쌓으면 해상도가 줄어들기도 전에 파라미터가 급증하여 오버핏(Overfitting)을 유발하므로 주기적인 Max Pooling의 개입이 필연적임을 파고들었습니다. 또한, ResNet이 34층까지는 3×3 2개로 구성된 'Basic Block'을 쓰다가 50층부터 1×1 구조의 'Bottleneck Block'으로 패러다임을 전환하여 연산량(FLOPs)과 파라미터를 통제하는 현실적인 제약과 실험적 근거를 분석했습니다. 특히 고차원 세부 정보(High-level Semantic)를 가장 가성비 좋게 추출할 수 있는 conv4_x 스테이지를 집중 반복([3, 4, 23, 3])하는 구조적 설계를 추적했습니다.

### 3. Knowledge Distillation의 수식적 비대칭성: 왜 Student의 예측값에만 log_softmax를 적용하는가?
지식의 증류(Knowledge Distillation) 코드를 구현하다 보면 KL 다이버전스(Kullback-Leibler Divergence) 손실 함수를 사용할 때 묘한 비대칭성을 발견하게 됩니다. 수식을 전개하면 Student 분포 $Q(i)$ 측에는 반드시 로그(log)가 씌워진 값이 들어가야 합니다. 일반 softmax 함수는 지수 함수를 쓰기 때문에 아주 작은 확률 값을 0으로 수렴시켜 외부에서 로그를 취할 때 수치적 불안정성($-\infty$ Underflow)을 유발하지만, 파이토치의 `log_softmax`는 내부적으로 'Log-Sum-Exp 트릭'을 사용하여 확률이 극도로 낮아져도 안정적인 로그 음수 값을 보장하고 그래디언트 폭발(NaN)을 막아주는 강력한 방어벽 역할을 수행함을 수리적으로 규명했습니다.

### 4. CAM (Class Activation Map)의 정당성: AvgPool로 뭉개진 채널별 숫자 1개가 어떻게 정보 비교의 근거가 되는가?
특징 맵(Feature Map)의 거대한 공간 정보를 GAP(Global Average Pooling)을 통해 단 하나의 숫자로 압축해 버렸음에도 어떻게 원본 공간의 중요 위치를 복원할 수 있는지 추적했습니다. 이는 선형 대수의 '시그마($\sum$)의 교환법칙'에 의해 GAP를 수행한 후 가중치와 선형 결합을 하나, 선형 결합을 먼저 한 후 공간 평균을 구하나 수학적으로 완벽하게 등가(Identity)이기 때문입니다. GAP는 공간 정보를 파괴하는 것이 아니라, 각 채널이 가진 공간적 고유 패턴을 유지한 채 '순수한 중요도 지분'만 스크리닝하여 가중치 $w_k^c$를 학습할 수 있도록 통로를 열어준 신의 한 수임을 수식적 전개와 실험으로 증명했습니다.

---

## 핵심 실증 실험 (Experiments & Results)

### 실험 1. Dropout과 Batch Normalization 배치 순서에 따른 역전파 및 출력 분포 안정성 검증
- **파일명:** `01_bn_dropout_interference.py`
- **실험 목적 (Why):** Dropout이 BN 앞에 위치할 때 발생하는 미니배치 통계량 교란 현상과 이로 인한 Inference Shift(Train vs Eval 모드 간의 출력값 분포 붕괴) 및 그래디언트 불안정성을 눈으로 확인합니다.
- **실험 방법 (How):** `BadBlock(Conv->Dropout->BN->ReLU)`과 `GoodBlock(Conv->BN->ReLU->Dropout)`을 구축하고 가상의 텐서 유입 환경에서 100번의 Iteration 동안 `running_var`의 추이를 추적하고, Train/Eval 모드 변환 시 출력값의 히스토그램 분포를 대조 시각화합니다.
- **실험 결과 (Result):** Bad 구조는 `running_var`가 정상 스케일(1.0)에서 크게 이탈(0.6112)하고 Eval 모드 전환 시 분포가 처참히 무너지는 반면, Good 구조는 일정한 스케일 유지 및 완벽하게 일치하는 안정적 스펙트럼을 증명했습니다.

### 실험 2. ResNet Layer 수 및 Bottleneck 구조 도입에 따른 연산 효율성 진단
- **파일명:** `02_resnet_bottleneck_flops.py`
- **실험 목적 (Why):** 깊은 신경망(50층 이상)에서 연산 스케일을 제어하기 위해 Basic Block 대신 Bottleneck Block(1x1 -> 3x3 -> 1x1) 구조적 패러다임을 전환해야 하는 당위성을 정량적으로 확인합니다.
- **실험 방법 (How):** 동일한 입력/출력 채널 환경에서 층을 깊게 쌓아 올릴 때 발생하는 파라미터 수와 FLOPs 증가 추이를 비교 평가합니다.
- **실험 결과 (Result):** 50층 이상의 딥 네트워크에서 Bottleneck 구조를 채택함으로써 연산 비용과 파라미터 폭발을 방지하고 고차원 시맨틱 특징 추출 효율을 가성비 좋게 끌어올릴 수 있음을 실측했습니다.

### 실험 3. KL-Divergence 입력에 따른 수치적 안정성(Numerical Instability) 테스트
- **파일명:** `03_kd_numerical_stability.py`
- **실험 목적 (Why):** Knowledge Distillation 과정에서 Student의 로짓에 일반 `F.softmax()` 적용 후 외부 `torch.log()`를 취하는 방식과 `F.log_softmax()`를 직접 사용하는 방식의 수치적 계산 안정성을 비교합니다.
- **실험 방법 (How):** 극단적인 Underflow/Overflow를 유발하는 큰/작은 로짓 값([[1000.0, -1000.0, 0.0]])을 생성하고, 두 가지 경로를 통해 계산된 최종 KLDivLoss 값을 비교 측정합니다.
- **실험 결과 (Result):** 직관적인 방식(Softmax + log)은 확률이 0으로 수렴하며 `-inf`가 발생해 최종 Loss가 `inf`로 폭발(NaN 유발)했으나, `log_softmax` 방식을 채택했을 때는 Log-Sum-Exp 트릭을 통해 극단적 도메인에서도 아주 안정적인 실숫값 손실을 반환함을 검증했습니다.

### 실험 4. 시그마 교환 법칙을 통한 CAM 메커니즘 등가성 수학적 증명
- **파일명:** `04_cam_mathematical_identity.py`
- **실험 목적 (Why):** GAP로 공간 정보를 뭉개버린 채널별 기여도 숫자가 선형 대수의 교환 법칙에 의해 어떻게 원본 공간 가중치 맵(CAM)과 수식적으로 완벽히 동일한지 정량 검증합니다.
- **실험 방법 (How):** 임의의 특징 맵(B=1, C=4, H=3, W=3)과 클래스 가중치를 선언한 뒤, 경로 A(GAP 선수행 후 가중치 결합)와 경로 B(가중치 결합 선수행 후 공간 평균)의 연산 결과를 도출하여 오차를 측정합니다.
- **실험 결과 (Result):** 두 연산 경로의 최종 Score 결과가 소수점 끝자리까지 정확히 일치하며 절대 오차가 `0.00000000e+00`임을 기록하여, GAP가 정보 손실 없이 공간 활성 정보를 완벽하게 보존함을 증명했습니다.

---

## 핵심 데이터 수치 요약 (Analysis)

| 측정 및 분석 대상 | 데이터 스케일 / 환경 | 평가 및 실측 지표 | 성능 및 기하학적 민감도 진단 |
| :--- | :--- | :--- | :--- |
| **BN-Dropout 배치 순서** | 32 배치 크기 / 32 채널 | Conv 레이어 Grad Norm / Running Var | Bad 구조는 Var가 0.6112로 이탈하며 분포 붕괴, Good 구조는 1.0 안팎의 안정적 스케일 및 분포 사수 |
| **ResNet 블록 패러다임** | 50층+ 확장 시나리오 | 파라미터 수 및 연산량 (FLOPs) | Basic Block 대비 Bottleneck 구조 적용 시 고차원 conv4_x 반복 연산량 가성비 극대화 확인 |
| **KD 수치적 안정성** | 극단적 로짓 (`1000.0`, `-1000.0`) | 계산된 KLDivLoss 값 및 inf 여부 | Naive 방식은 Underflow로 `inf` 손실 유발, `log_softmax`는 Log-Sum-Exp 트릭으로 안정적 손실 반환 |
| **CAM 메커니즘 등가성** | 1x4x3x3 가상 특징 맵 텐서 | 경로 A vs 경로 B 간의 절대 오차 (Absolute Error) | 두 경로의 최종 Score가 일치하며 절대 오차 `0.00000000e+00` 기록 (수학적 동등성 보존 완료) |

---

## 기술 스택 및 라이브러리

- **Language:** Python 3.10+
- **Deep Learning Framework:** PyTorch (`torch`, `torch.nn`, `torch.functional` as `F`)
- **Data Science & Math:** NumPy
- **Visualization:** Matplotlib

---

## 핵심 소스코드 아키텍처

### 1. KL-Divergence 입력 수치 안정성 검증 모듈 (`03_kd_numerical_stability.py`)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Underflow/Overflow를 유발하는 극단적인 상황의 가상 로짓 연출
student_logits = torch.tensor([[1000.0, -1000.0, 0.0]], dtype=torch.float32)
teacher_logits = torch.tensor([[10.0, -5.0, 2.0]], dtype=torch.float32)
T = 2.0  # Temperature

p_teacher = F.softmax(teacher_logits / T, dim=1)
loss_fn = nn.KLDivLoss(reduction='batchmean')

# 방법 A: 직관적이지만 위험한 방식 (Underflow로 인해 inf 폭발 위험)
q_student_naive = F.softmax(student_logits / T, dim=1)
log_q_naive = torch.log(q_student_naive)
loss_a = loss_fn(log_q_naive, p_teacher)  # 결과: inf

# 방법 B: 파이토치 권장 방식 (Log-Sum-Exp 트릭을 통한 수치적 안정성 확보)
log_q_stable = F.log_softmax(student_logits / T, dim=1)
loss_b = loss_fn(log_q_stable, p_teacher)  # 결과: 안정적인 실숫값 반환
