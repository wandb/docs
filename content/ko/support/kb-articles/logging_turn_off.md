---
title: 로그를 끄려면 어떻게 하나요?
menu:
  support:
    identifier: ko-support-kb-articles-logging_turn_off
support:
- 로그
toc_hide: true
type: docs
url: /support/:filename
---

`wandb offline` 코맨드는 환경 변수 `WANDB_MODE=offline`을 설정하여 데이터가 원격 W&B 서버와 동기화되지 않도록 합니다. 이 동작은 모든 Projects에 영향을 주며, W&B 서버로의 데이터 로그를 중단합니다.

경고 메시지를 숨기려면 다음 코드를 사용하세요:

```python
import logging

logger = logging.getLogger("wandb")
logger.setLevel(logging.WARNING)
```