# [Hands-on ML & DL] 02. Recurrent Neural Network & Attention Paradigm Optimization

《핸즈온 머신러닝(15, 16장)》 및 《딥러닝의 정석(6장)》을 깊게 탐독하는 과정에서 마주한 구조적 의문들을 정리하고, 순환 신경망(RNN) 계열의 핵심 메커니즘을 직접 실험 및 증명한 보고서입니다. 단순한 기성 관례(Convention)나 API 블랙박스로 치부되던 게이트형 RNN(LSTM, GRU)의 내부 가중치 자율 분화 원리부터 PyTorch 고유의 텐서 기하학적 메모리 배치 규칙, 그리고 트랜스포머의 근간이 되는 Self-Attention의 연산 복잡도를 파괴하는 Effective Attention의 수학적 펀더멘탈까지 직접 파고들며 최적화의 원리를 본질적으로 해체해 본 트러블슈팅 리포트입니다.

---

## 핵심 실험 및 트러블슈팅 요약

> 상세한 이론적 배경과 분석 리포트는 📄 [기술 블로그 포스팅/벨로그](https://velog.io) 에 별도 정리되어 있습니다.

### 1. LSTM/GRU 게이트의 자율성과 역할 분담: 왜 명시적 설정 없이도 각 게이트가 제 역할을 찾아가는가?
LSTM 논문을 보면 포겟 게이트($f_t$), 입력 게이트($i_t$), 출력 게이트($o_t$)가 각각 '망각', '기억', '출력'이라는 명확한 도메인 역할을 부여받았습니다. 하지만 코드를 구현할 때 우리는 어떤 제약 조건이나 가이드라인도 주지 않습니다. 본 리포트에서는 초기 무작위 상태의 가중치들이 오직 수식 레이아웃이 가지는 위상(Topology)적 차이와 역전파되는 그라디언트의 흐름에 의해 대칭성이 깨지고(Symmetry Breaking), 엔트로피를 낮추는 방향으로 스스로의 역할을 자율 분담해 나감을 실증했습니다.

### 2. PyTorch LSTM 텐서의 기하학: 왜 Output과 Hidden State의 차원은 불일치하는가?
PyTorch LSTM의 결과물인 `output`과 `h_n`은 기괴한 차원 불일치를 보입니다. `output`은 `[seq_len, batch_size, num_directions * hidden_size]`인 반면, `h_n`은 `[num_layers * num_directions, batch_size, hidden_size]` 구조를 가집니다. 이는 두 출력이 존재하는 목적의 차이를 반영한 기하학적 설계입니다. 시계열 흐름 전체를 보존하는 `output`과 달리, `h_n`은 순환 신경망의 전체 레이어 컨텍스트를 온전히 전달하기 위해 독립적인 은닉 벡터들을 서랍장처럼 차곡차곡 쌓아 관리하는 방식임을 텐서 슬라이싱 메모리 매핑을 통해 역추적 및 물리적으로 증명했습니다.

### 3. Self-Attention vs Effective Attention: 결합법칙을 이용한 시퀀스 길이($N$)의 저주 탈출
Transformer의 핵심인 Standard Self-Attention은 $Q$와 $K^T$를 먼저 곱해 $N \times N$ 크기의 거대한 어텐션 맵을 생성하므로 시퀀스 길이 $N$에 대해 $O(N^2)$의 시공간 복잡도를 가집니다. 반면 Effective Attention은 수학적 정규화(Softmax)의 위치를 살짝 비틂으로써 행렬 곱셈의 결합법칙($A(BC) = (AB)C$)을 극적으로 활용합니다. $K^T$와 $V$를 먼저 연산해 $D \times D$ 크기의 은닉 컨텍스트 행렬을 생성함으로써, $N \times N$ 행렬 생성을 원천 차단하고 복잡도를 $O(N)$으로 선형 감소시키는 최적화 패러다임을 검증했습니다.

---

## 핵심 실증 실험 (Experiments & Results)

### 실험 1. LSTM 게이트 가중치 자율 분화 및 그라디언트 수렴성 검증
* **파일명**: `exp01_lstm_gate_autonomy_tracker.py`
* **실험 목적 (Why)**: 동일한 손실(Loss)을 넘겨받았음에도, 입력 데이터의 시점(과거 vs 현재)과 수식 구조적 위상 차이에 의해 `Input Gate`와 `Forget Gate`가 서로 다른 부호와 크기의 그라디언트를 흡수하며 자율적으로 역할을 분화해 나가는 현상을 눈으로 확인합니다.
* **실험 방법 (How)**: 과거 정보($T=0$)와 현재 정보($T=-1$)를 동시에 참조해야만 해결할 수 있는 XOR 형태의 시계열 데이터셋을 구축합니다. 단층 LSTM의 가중치 행렬 그라디언트를 가로채어(`weight_ih_l0.grad`) 청크 분할한 뒤, 에폭 진행에 따른 개별 게이트 가중치들의 고유한 수렴 추이를 추적합니다.
* **실험 결과 (Result)**: 초기에는 무작위 상태이던 그라디언트가 학습이 진행됨에 따라 `Input Gate`와 `Forget Gate`에서 완전히 다른 부호와 크기 스케일(Symmetry Breaking)로 분화되어 각자의 도메인 역할을 독립적으로 학습해 나감을 실측했습니다.

### 실험 2. PyTorch LSTM 내부 차원 기하학의 물리적 증명
* **파일명**: `exp02_pytorch_lstm_dim_geometry.py`
* **실험 목적 (Why)**: 다층(Multi-layer) 및 양방향(Bidirectional) LSTM 구동 시 뱉어내는 `output`과 `h_n` 간의 차원 불일치가 단순한 형태의 차이를 넘어, 메모리 상에서 실제 어떤 타임스텝 및 레이어와 정교하게 매핑되는지 물리적 일치성을 검증합니다.
* **실험 방법 (How)**: 2개 레이어와 양방향 구조를 강제한 LSTM에 임의의 텐서를 입력하고 순전파를 수행합니다. 정방향 연산의 최종 상태는 시퀀스의 끝($T=-1$)에, 역방향 연산의 최종 상태는 시퀀스의 시작($T=0$)에 기록된다는 양방향 RNN의 시간 축 붕괴 가설을 기반으로 두 텐서를 슬라이싱하여 비교합니다.
* **실험 결과 (Result)**: `h_n` 최상위 레이어의 정방향 메모리가 `output[-1, :, :hidden_size]`와 소수점 끝자리까지 정확히 일치하며, 역방향 메모리 역시 `output[0, :, hidden_size:]`와 절대 오차 `0.0`으로 매핑됨을 기록하여 구조적 인덱싱 규칙을 완벽히 증명했습니다.

### 실험 3. Standard Self-Attention vs Effective Attention 메모리 점유 및 연산 프로파일러
* **파일명**: `exp03_attention_memory_profiler.py`
* **실험 목적 (Why)**: 시퀀스 길이 $N$이 은닉 차원 $D$보다 압도적으로 긴 가혹 환경($N \gg D$)에서 표준 셀프 어텐션이 겪는 $O(N^2)$ 메모리 폭발 현상과 이펙티브 어텐션의 결합법칙 우회를 통한 $O(N)$ 최적화 성능을 실제 물리 메모리 단위로 대조 평가합니다.
* **실험 방법 (How)**: 시퀀스 길이 $N=15,000$, 차원 $D=64$의 가상 텐서 환경을 구축한 후 PyTorch의 CUDA 가속 프로파일러(`torch.cuda.memory_allocated()`)를 활용해 각 메커니즘이 소모하는 순수 추가 메모리량(MB)과 연산 동기화 시간(Latency)을 정밀 실측합니다.
* **실험 결과 (Result)**: Standard 방식은 거대한 $N \times N$ 행렬 생성으로 인해 메모리 점유가 폭발한 반면, Effective 방식은 $D \times D$ 컨텍스트 매트릭스를 선제 성형함으로써 메모리 사용량을 획기적으로 절감하고 연산 속도를 가성비 좋게 끌어올릴 수 있음을 실측 증명했습니다.

---

## 핵심 데이터 수치 요약 (Analysis)

| 측정 및 분석 대상 | 데이터 스케일 / 환경 | 평가 및 실측 지표 | 성능 및 기하학적 민감도 진단 |
| :--- | :--- | :--- | :--- |
| **LSTM 게이트 자율 분화** | 배치 64 / 길이 5 / Hidden 4 | 개별 게이트 가중치 Grad 분화도 | 학습 초기 대칭 상태 파괴 후, Input과 Forget 게이트가 서로 상반된 부호 및 스케일로 완전히 독립 수렴 확인 |
| **LSTM 내부 차원 매핑** | 2 Layers / Bidirectional | 두 텐서 간의 절대 오차 (Absolute Error) | 정방향($T=-1$) 및 역방향($T=0$) 슬라이싱 매칭 결과 오차 `0.000000e+00` 기록 (물리적 등가성 사수) |
| **어텐션 패러다임 시프트** | 시퀀스 15,000 / 차원 64 | GPU 추가 할당 메모리 (MB) | Standard 대비 Effective 구조 적용 시 $O(N^2)$ 메모리 병목을 완벽히 회피하며 압도적인 메모리 절감 효율 달성 |

---

## 기술 스택 및 라이브러리

* **Language**: Python 3.10+
* **Deep Learning Framework**: PyTorch (`torch`, `torch.nn`, `torch.functional as F`)
* **Hardware Acceleration**: CUDA Enabler
* **Data Science & Benchmarking**: `time`, `math`

---

## 핵심 소스코드 아키텍처

### 1. PyTorch LSTM 내부 차원 기하학 물리적 증명 모듈 (`exp02_pytorch_lstm_dim_geometry.py`)

```python
import torch
import torch.nn as nn

# 1. 하이퍼파라미터 정의 (다층 + 양방향 모델 구조 강제)
seq_len = 5
batch_size = 2
input_size = 4
hidden_size = 3
num_layers = 2

# Bidirectional 모델 선언
bi_lstm = nn.LSTM(
    input_size=input_size, 
    hidden_size=hidden_size, 
    num_layers=num_layers, 
    bidirectional=True
)

# 더미 시퀀스 데이터 생성 (seq_len, batch_size, input_size)
X = torch.randn(seq_len, batch_size, input_size)

# 순전파 실행
output, (h_n, c_n) = bi_lstm(X)

print("=== [실험 2] 텐서 기하학적 슬라이싱 매치 ===")
print(f"Output Shape : {list(output.shape)}")  # [5, 2, 6] -> 6 = 2(방향) * 3(hidden)
print(f"H_n Shape    : {list(h_n.shape)}")     # [4, 2, 3] -> 4 = 2(레이어) * 2(방향)

print("\n--- 수학적/물리적 일치성 검증 ---")

# 가설 1: output의 최후 타임스텝(T=-1) 내 정방향(Forward) 벡터는 
#        h_n의 최상위 레이어(Layer 2)의 정방향 인덱스 메모리와 완벽히 일치해야 한다.
top_layer_forward_hn = h_n[2, :, :] 
last_timestep_output_forward = output[-1, :, :hidden_size]

is_matching = torch.allclose(top_layer_forward_hn, last_timestep_output_forward, atol=1e-6)
print(f"▶ 최상위 레이어 [정방향] 일치 여부: {is_matching}")

# 가설 2: 역방향(Backward)의 최종 상태는 시퀀스의 시작점(T=0)에 기록된다.
top_layer_backward_hn = h_n[3, :, :]
first_timestep_output_backward = output[0, :, hidden_size:]

is_backward_matching = torch.allclose(top_layer_backward_hn, first_timestep_output_backward, atol=1e-6)
print(f"▶ 최상위 레이어 [역방향] 일치 여부: {is_backward_matching}")
