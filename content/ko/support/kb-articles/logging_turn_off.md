---
title: How do I turn off logging?
menu:
  support:
    identifier: ko-support-kb-articles-logging_turn_off
support:
- logs
toc_hide: true
type: docs
url: /ko/support/:filename
---

`wandb offline` 코맨드는 환경 변수 `WANDB_MODE=offline`을 설정하여, 데이터가 원격 W&B 서버에 동기화되는 것을 막습니다. 이 동작은 모든 프로젝트에 영향을 미치며, W&B 서버로의 데이터 로깅을 중단합니다.

경고 메시지를 표시하지 않으려면 다음 코드를 사용하세요.

```python
import logging

logger = logging.getLogger("wandb")
logger.setLevel(logging.WARNING)
```