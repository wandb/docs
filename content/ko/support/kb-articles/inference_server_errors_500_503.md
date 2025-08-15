---
title: W&B Inference에서 server 오류(500, 503)가 발생할 때 어떻게 해결할 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-inference_server_errors_500_503
support:
- 추론
toc_hide: true
type: docs
url: /support/:filename
---

서버 오류는 W&B Inference 서비스에서 일시적으로 발생하는 문제를 나타냅니다.

## 오류 유형

### 500 - Internal Server Error
**메시지:** "The server had an error while processing your request"

서버 측에서 발생한 일시적인 내부 오류입니다.

### 503 - Service Overloaded
**메시지:** "The engine is currently overloaded, please try again later"

서비스에 많은 트래픽이 몰리고 있는 상태입니다.

## 서버 오류 처리 방법

1. **잠시 기다린 후 재시도하기**
   - 500 오류: 30-60초 대기
   - 503 오류: 60-120초 대기

2. **지수적 백오프(exponential backoff) 사용**
   ```python
   import time
   import openai
   
   def call_with_retry(client, messages, model, max_retries=5):
       for attempt in range(max_retries):
           try:
               return client.chat.completions.create(
                   model=model,
                   messages=messages
               )
           except Exception as e:
               if "500" in str(e) or "503" in str(e):
                   if attempt < max_retries - 1:
                       wait_time = min(60, (2 ** attempt))
                       time.sleep(wait_time)   # 지정된 시간만큼 대기 후 재시도
                   else:
                       raise    # 재시도 최대 횟수 도달시 예외 발생
               else:
                   raise    # 다른 예외 발생시 예외 그대로 전달
   ```

3. **적절한 타임아웃 설정**
   - HTTP 클라이언트의 타임아웃 값을 늘리세요
   - 비동기(async) 연산을 고려하면 더 원활한 처리가 가능합니다

## 고객 지원팀에 문의해야 할 때

다음의 경우 지원팀에 문의하세요:
- 오류가 10분 이상 지속되는 경우
- 특정 시간마다 반복적으로 실패하는 패턴이 보일 때
- 오류 메시지에 추가적인 세부 정보가 포함되어 있을 때

아래 정보를 함께 제공해 주세요:
- 오류 메시지와 코드
- 오류 발생 시간
- 사용한 코드조각(API 키는 삭제)
- W&B entity 및 project 이름