---
title: How can I resume a sweep using Python code?
menu:
  support:
    identifier: ko-support-kb-articles-resume_sweep_using_python_code
support:
- sweeps
- python
toc_hide: true
type: docs
url: /ko/support/:filename
---

스윕을 재개하려면 `sweep_id`를 `wandb.agent()` 함수에 전달하세요.

```python
import wandb

sweep_id = "your_sweep_id"

def train():
    # 트레이닝 코드
    pass

wandb.agent(sweep_id=sweep_id, function=train)
```