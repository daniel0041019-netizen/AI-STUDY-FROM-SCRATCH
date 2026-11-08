import asyncio
from openai import AsyncOpenAI, RateLimitError, APIConnectionError

# 비동기 전용 클라이언트 초기화
async_client = AsyncOpenAI()

async def fetch_with_exponential_backoff(prompt: str, max_retries: int = 4) -> str:
    """RateLimitError 등 네트워크 I/O 장애에 대응하는 방어적 재시도 지수 백오프 패턴"""
    delay = 1.0  # 초기 대기 시간 (1초)
    
    for attempt in range(max_retries):
        try:
            response = await async_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            return response.choices[0].message.content
        
        except (RateLimitError, APIConnectionError) as e:
            if attempt == max_retries - 1:
                print(f"[ERROR] 최대 재시도 횟수 초과: {prompt}")
                raise e
            
            print(f"[WARN] 재시도 ({attempt + 1}/{max_retries}) - {delay}초 후 대기 후 재호출...")
            await asyncio.sleep(delay)
            delay *= 2.0  # 지수적으로 대기 시간을 2배 늘림 (1s -> 2s -> 4s -> 8s)

async def run_async_pipeline():
    prompts = [
        "LLM 애플리케이션에서 비동기 I/O 처리가 필수적인 이유 1가지",
        "Pydantic 기반 Structured Outputs의 최대 강점 1가지",
        "시스템 프롬프트를 마크다운으로 작성해야 하는 이유 1가지",
        "동적 Few-shot 주입 기법이 가져다주는 이점 1가지"
    ]

    print("=== [Part 4] asyncio 병렬 요청 시작 ===")
    
    # asyncio.gather를 통한 수십~수백 개의 동시 LLM 비동기 병렬 요청
    tasks = [fetch_with_exponential_backoff(p) for p in prompts]
    results = await asyncio.gather(*tasks)

    for i, res in enumerate(results, 1):
        print(f"\n[결과 {i}]\n{res.strip()}")

if __name__ == "__main__":
    asyncio.run(run_async_pipeline())
