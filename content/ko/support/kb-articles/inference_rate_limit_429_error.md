---
title: W&B Inference 사용 시 왜 속도 제한 오류(429)가 발생하나요?
menu:
  support:
    identifier: ko-support-kb-articles-inference_rate_limit_429_error
support:
- 추론
toc_hide: true
type: docs
url: /support/:filename
---

Rate limit 오류(429)는 동시 처리 한도를 초과하거나 크레딧이 부족할 때 발생합니다.

## 429 오류 유형

### 동시성 한도 도달
**오류:** "Concurrency limit reached for requests"

**해결 방법:**
- 동시에 보내는 요청 수를 줄이세요
- 요청 사이에 지연을 추가하세요
- 지수 백오프를 구현하세요
- 참고: Rate limit은 각 W&B Project 별로 적용됩니다

### 할당량 초과
**오류:** "You exceeded your current quota, please check your plan and billing details"

**해결 방법:**
- W&B Billing 페이지에서 크레딧 잔액을 확인하세요
- 더 많은 크레딧을 구매하거나 플랜을 업그레이드하세요
- 지원팀에 한도 상향을 요청하세요

### 개인 계정 제한
**오류:** "W&B Inference isn't available for personal accounts"

**해결 방법:**
- 개인이 아닌 계정으로 전환하세요
- Team을 생성해서 W&B Inference에 엑세스하세요
- Personal Entities는 2024년 5월에 더 이상 지원되지 않습니다

## Rate limit을 피하는 모범 사례

1. **지수 백오프를 활용한 재시도 로직 구현:**
   ```python
   import time
   
   def retry_with_backoff(func, max_retries=3):
       for i in range(max_retries):
           try:
               return func()
           except Exception as e:
               # "429" 오류 시 재시도, 최대 재시도 횟수까지 대기 시간 증가
               if "429" in str(e) and i < max_retries - 1:
                   time.sleep(2 ** i)
               else:
                   raise
   ```

2. **병렬 요청 대신 배치 처리 사용**

3. **W&B Billing 페이지에서 사용량을 모니터링하세요**

## 기본 지출 한도

- **Pro 계정:** 월 $6,000
- **Enterprise 계정:** 연 $700,000

한도 조정이 필요하면 계정 담당자나 지원팀에 문의하세요.