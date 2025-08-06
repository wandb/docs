---
title: Python 코드로 스윕을 어떻게 재개할 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-resume_sweep_using_python_code
support:
- 스윕
- 파이썬
toc_hide: true
type: docs
url: /support/:filename
---

스윕을 재개하려면 `sweep_id` 를 `wandb.agent()` 함수에 전달하세요.

```python
import wandb

sweep_id = "your_sweep_id"

def train():
    # 트레이닝 코드 작성
    pass

wandb.agent(sweep_id=sweep_id, function=train)
```