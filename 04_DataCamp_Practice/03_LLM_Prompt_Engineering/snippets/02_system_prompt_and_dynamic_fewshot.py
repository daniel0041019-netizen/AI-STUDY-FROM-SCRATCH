from openai import OpenAI

client = OpenAI()

# 1. 마크다운으로 계층화된 프로페셔널 시스템 프롬프트 템플릿 (4대 핵심 블록)
SYSTEM_PROMPT_TEMPLATE = """
## Role
당신은 기술 지원 전문 데이터 분석가입니다.

## Objective
사용자의 문의 내역을 바탕으로 카테고리를 분류하고 핵심 요구사항을 1줄로 요약하세요.

## Constraints
- 반드시 한국어로 작성할 것.
- 감정적인 표현은 배제하고 사실 위주로 설명할 것.
- 50자 이내로 간결히 요약할 check constraints 준수.

## Output Format
카테고리: [분류명]
요약: [요약 내용]

---
## Examples (Context Injected)
{dynamic_examples}
"""

# DB/메모리에서 사용자 입력과 가장 유사한 예시 2~3개를 동적으로 추출하는 모의 함수
def retrieve_dynamic_examples(user_query: str) -> str:
    # 실전에서는 RAG / Vector DB 기반 검색 알고리즘 적용
    if "결제" in user_query or "환불" in user_query:
        return """
[예시 1]
입력: "지난주에 결제했는데 중복 차감되었어요. 환불해주세요."
출력: 
카테고리: 결제/환불
요약: 중복 결제 건에 대한 환불 요청
"""
    return """
[예시 1]
입력: "비밀번호를 잊어버려서 로그인이 안 됩니다."
출력: 
카테고리: 계정/인증
요약: 계정 비밀번호 재설정 문의
"""

def run_dynamic_fewshot(user_input: str):
    # 동적 예시 주입 (Dynamic Example Injection)
    examples = retrieve_dynamic_examples(user_input)
    formatted_system_prompt = SYSTEM_PROMPT_TEMPLATE.format(dynamic_examples=examples)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": formatted_system_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=0.1
    )
    
    print("=== [Part 2] Dynamic Few-shot 적용 출력 ===")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    run_dynamic_fewshot("카드 결제 내역서 발급은 어디서 하나요?")
