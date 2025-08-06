---
title: W&B Inference에서 Invalid Authentication (401) 오류를 어떻게 해결하나요?
menu:
  support:
    identifier: ko-support-kb-articles-inference_authentication_401_error
support:
- 추론
toc_hide: true
type: docs
url: /support/:filename
---

401 Invalid Authentication 에러는 API 키가 유효하지 않거나 W&B 프로젝트 엔터티/이름이 올바르지 않을 때 발생합니다.

## API 키 확인

1. [https://wandb.ai/authorize](https://wandb.ai/authorize)에서 새로운 API 키를 발급받으세요.
2. 복사할 때 불필요한 공백이나 누락된 문자가 없는지 확인하세요.
3. API 키를 안전하게 보관하세요.

## 프로젝트 설정 확인

프로젝트가 `<your-team>/<your-project>` 형식으로 올바르게 입력되어 있는지 확인하세요.

**Python 예시:**
```python
client = openai.OpenAI(
    base_url='https://api.inference.wandb.ai/v1',
    api_key="<your-api-key>",
    project="<your-team>/<your-project>",  # 반드시 W&B 팀과 프로젝트 이름과 일치해야 합니다
)
```

**Bash 예시:**
```bash
curl https://api.inference.wandb.ai/v1/chat/completions \
  -H "Authorization: Bearer <your-api-key>" \
  -H "OpenAI-Project: <your-team>/<your-project>"
```

## 자주 발생하는 실수

- 개인 Entity 대신 팀 이름 사용
- 팀 또는 프로젝트 이름 오타
- 팀과 프로젝트 사이 슬래시(`/`) 누락
- 만료되었거나 삭제된 API 키 사용

## 아직도 문제가 있나요?

- 해당 팀과 프로젝트가 W&B 계정 내에 존재하는지 확인하세요.
- 지정한 팀에 엑세스 권한이 있는지 확인하세요.
- 현재 API 키가 동작하지 않으면 새 API 키를 발급받아 시도해보세요.
