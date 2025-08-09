---
title: 왜 내 개인 계정에서는 W&B Inference를 사용할 수 없나요?
menu:
  support:
    identifier: ko-support-kb-articles-inference_personal_account_limitation
support:
- 추론
toc_hide: true
type: docs
url: /support/:filename
---

개인 계정은 W&B Inference 를 지원하지 않습니다. 다음과 같은 429 에러 메시지가 표시됩니다: "W&B Inference isn't available for personal accounts. Please switch to a non-personal account to access W&B Inference."

## 배경

Personal Entities 는 2024년 5월에 지원이 중단되었습니다. 이 변경 사항은 여전히 개인 entity 를 사용하는 기존 계정들에만 영향을 줍니다.

## W&B Inference 엑세스 방법

### Team 생성하기

1. W&B 계정에 로그인하세요.
2. 우측 상단의 프로필 아이콘을 클릭하세요.
3. "Create new team"을 선택하세요.
4. 팀 이름을 정하세요.
5. W&B Inference 요청에 이 팀을 사용하세요.

### 코드 업데이트

personal entity 를 team 으로 변경하세요:

**변경 전 (동작하지 않음):**
```python
project="your-username/project-name"  # personal entity
```

**변경 후 (동작함):**
```python
project="your-team/project-name"  # team entity
```

## Teams 사용의 장점

- W&B Inference 엑세스 가능
- 더 나은 협업 기능
- 공유 Projects 및 리소스
- 팀 기반 결제 및 사용량 추적

## 도움이 필요하신가요?

팀 생성이나 개인 계정에서 전환하는데 문제가 있다면, W&B 지원팀에 문의해 도움을 받으세요.