---
title: How do I save code?‌
menu:
  support:
    identifier: ko-support-save_code‌
tags:
- artifacts
toc_hide: true
type: docs
---

`wandb.init`에서 `save_code=True`를 사용하여 run을 시작하는 주요 스크립트 또는 노트북을 저장합니다. run에 대한 모든 코드를 저장하려면 Artifacts로 코드 버전을 관리하세요. 다음 예제는 이 프로세스를 보여줍니다.

```python
code_artifact = wandb.Artifact(type="code")
code_artifact.add_file("./train.py")
wandb.log_artifact(code_artifact)
```