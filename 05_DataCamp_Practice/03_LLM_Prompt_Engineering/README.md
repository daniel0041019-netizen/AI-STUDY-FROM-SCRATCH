# AI & LLM Engineering Summary

> **Core Philosophy:** OpenAI API의 기초 활용부터 시스템 프롬프트 구조화, 동적 Few-shot, Pydantic 기반 Structured Outputs, Function Calling, 그리고 asyncio 비동기 파이프라인 최적화 패턴 정리

---

## 1. OpenAI API 기초 및 인컨텍스트 러닝

### 기초 개념
* **엔지니어링 진입점:** 모델 및 서비스에 진입하는 상황에 따른 진입점 (예: 채팅 완성 엔드포인트 `client.chat.completions.create()`)
* **비밀 키 (API Key):** 오픈AI 환경설정 시 내 계정으로 API 서비스를 호출할 때 신원을 증명하는 필수 자격 증명
* **인컨텍스트 러닝 (In-Context Learning):** 모델의 가중치를 고동결(Frozen)한 상태로 유지하며, 프롬프트 내부의 지시와 예시를 통해 모델의 사고 방식을 실시간으로 재구성하는 기법

### 역할별 메시지 구조 (`messages`)
* **`system`:** 모델의 전체적인 페르소나, 핵심 목적, 제약 조건을 규정하는 중요한 템플릿 서식
* **`assistant`:** 인컨텍스트 러닝을 위한 모범 응답 예시 대화
* **`user`:** 새 입력에 필요한 컨텍스트 및 사용자 질의

### 주요 파라미터 제어
* **`temperature`:** 출력의 변동성을 조절. 정형화된 데이터 추출이나 제약 조건 준수가 필수적인 태스크에서는 `0.1 ~ 0.3` 사이로 낮게 설정
* **`max_completion_tokens`:** 모델이 생성할 수 있는 최대 토큰 수 제한

---

## 2. Advanced Prompt Engineering: 구조화와 동적 주입

### 시스템 프롬프트 마크다운 구조화
단순한 평서문 지시는 모델이 우선순위를 망각하기 쉬우므로, 마크다운 헤더(`#`, `##`)와 구분선(`---`)을 활용하여 토큰 계층화 및 지시사항을 명확히 분리해야 합니다.

### 프로페셔널 시스템 프롬프트의 4대 핵심 블록
1. **`## Role` (역할 및 페르소나):** 도메인 전문가 지정을 통해 관련 지식과 어휘 분포 활성화
2. **`## Objective` (핵심 목적):** 완수해야 할 단 하나의 명확한 미션 정의
3. **`## Constraints` (제약 조건 및 가드레일):** 글자 수, 형식, 부정적 제약(Negative Constraints) 부여
4. **`## Output Format` (출력 형식):** 백엔드 후속 파싱을 위한 규격 강제

### Dynamic Example Injection (동적 예시 주입)
* **개념:** 모든 요청에 고정된 예시를 쓰는 대신, 현재 사용자 입력의 성격과 도메인에 가장 잘 어울리는 핵심 예시 2~3개를 데이터베이스나 메모리에서 골라 실시간으로 프롬프트에 주입하는 패턴
* **장점:** 토큰 효율성 극대화 및 환각(Hallucination) 현상 획기적 감소

---

## 3. Structured Outputs: Pydantic과 JSON 강제화

### 백엔드 파싱의 한계 극복
일반적인 LLM 호출은 마크다운 코드 블록이나 키 이름 변동 등 비정형 문자열을 반환하여 백엔드 파서를 멈추게(`JSONDecodeError`) 만듭니다. 이를 원천 차단하기 위해 OpenAI JSON Mode와 Pydantic Schema를 결합합니다.

### 비교: `json_object` vs `client.beta.chat.completions.parse()`
* **`response_format={"type": "json_object"}`:** 구문적 신뢰성(JSON 문법 준수)만 보장하며, 개발자가 직접 `json.loads()`로 수동 파싱 및 검증 필요
* **`client.beta.chat.completions.parse(response_format=PydanticModel)`:** 개발자가 정의한 Pydantic 스키마와 100% 일치하는 데이터 구조와 타입까지 완벽히 강제하며, 별도의 파싱 없이 `.parsed`로 즉시 사용 가능 

### Pydantic 활용 핵심 포인트
* **`Field(description=...)`:** 필드별 의미를 자연어로 가이드하여 추출 정확도 향상
* **`model_json_schema()`:** Pydantic 모델 구조를 OpenAI API가 요구하는 JSON Schema 규격으로 자동 변환
* **`Literal` (`typing.Literal`):** 변수나 필드에 오직 특정한 구체적인 값들만 들어오도록 제한

---

## 4. Function Calling / Tool Use 기초

### 동작 원리
1. **도구 등록 (Definition):** 시스템이 수행할 수 있는 함수 스키마(이름, 설명, 파라미터)를 LLM에게 사전 공유
2. **인자 추출 및 판단 (Model Inference):** 사용자의 자연어 요청을 분석해 적절한 도구를 선택하고 필요한 인자를 JSON 구조로 추출 (`tool_calls`)
3. **백엔드 실행 및 피드백 (Execution & Response):** 추출된 인자를 파이썬 함수에 전달해 실행한 뒤, 결과를 모델에게 피드백하여 최종 응답 생성

---

## 5. Python asyncio와 Async OpenAI Client 고속 파이프라인

### 비동기 프로그래밍 (`asyncio` & `AsyncOpenAI`)
* **동기 처리의 한계:** 순차적 대기로 인한 Blocking 발생 및 CPU 유휴 시간 증가
* **비동기 처리:** 네트워크 I/O 대기 시간 동안 제어권을 이벤트 루프에 양보하여, 수십·수백 개의 LLM 요청을 동시에 처리하고 먼저 끝나는 순서대로 취합 (`asyncio.gather(*tasks)`)

### API 장애 대응: Exponential Backoff (지수 백오프)
* **개념:** 대량의 비동기 요청 시 발생하는 `RateLimitError` 등에 대응하기 위해 재시도 대기 시간을 지수 함수 형태(1초, 2초, 4초, 8초 등)로 점진적으로 늘려가는 방어적 프로그래밍 기법 적용
