---
title: 스윕 로그를 로컬에 저장할 디렉터리를 어떻게 변경할 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-change_directory_sweep_logs_locally
support:
- Sweeps
toc_hide: true
type: docs
url: /support/:filename
---

`WANDB_DIR` 환경 변수를 설정하여 W&B run 데이터의 로그 디렉토리를 지정할 수 있습니다. 예를 들어:

```python
# 로그 디렉토리를 설정합니다.
os.environ["WANDB_DIR"] = os.path.abspath("your/directory")
```