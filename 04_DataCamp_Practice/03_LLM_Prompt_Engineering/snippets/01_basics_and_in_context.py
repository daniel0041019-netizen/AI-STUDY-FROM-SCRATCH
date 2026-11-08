import os
from openai import OpenAI

# 1. API 클라이언트 초기화 (API Key 필요)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def run_basic_chat():
    # 2. chat.completions.create() - 기본 엔드포인트 진입
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            # system: 페르소나 및 기본 제약 규정
            {"role": "system", "content": "너는 신중하고 정형화된 기술 명세 작성자이다."},
            
            # assistant & user: 인컨텍스트 러닝(In-Context Learning)을 위한 모범 대화 예시
            {"role": "user", "content": "Q: 인컨텍스트 러닝이란?"},
            {"role": "assistant", "content": "A: 모델의 가중치를 고동결(Frozen)한 채, 프롬프트 내부 예시로 사고방식을 재구성하는 기법입니다."},
            
            # user: 실제 사용자의 신규 입력 질의
            {"role": "user", "content": "Q: 파라미터 중 temperature의 역할을 설명해줘."}
        ],
        # 주요 파라미터 제어
        temperature=0.2,            # 낮은 변동성 (정수/정형 추출 작업 시 0.1 ~ 0.3 권장)
        max_completion_tokens=150   # 모델이 생성할 수 있는 최대 토큰 제한
    )
    
    print("=== [Part 1] 기본 completion 출력 ===")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    run_basic_chat()
