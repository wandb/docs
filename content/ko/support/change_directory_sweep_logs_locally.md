---
title: How can I change the directory my sweep logs to locally?
menu:
  support:
    identifier: ko-support-change_directory_sweep_logs_locally
tags:
- sweeps
toc_hide: true
type: docs
---

`WANDB_DIR` 환경 변수를 설정하여 W&B run 데이터의 로깅 디렉토리를 설정합니다. 예를 들면 다음과 같습니다.

```python
os.environ["WANDB_DIR"] = os.path.abspath("your/directory")
```
