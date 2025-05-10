---
title: How do I save code?‌
menu:
  support:
    identifier: ko-support-kb-articles-save_code‌
support:
- artifacts
toc_hide: true
type: docs
url: /ko/support/:filename
translationKey: save_code‌
---
`wandb.init`에서 `save_code=True`를 사용하면 run을 시작하는 메인 스크립트 또는 노트북이 저장됩니다. run에 대한 모든 코드를 저장하려면 Artifacts로 코드의 버전을 관리하세요. 다음 예제는 이 프로세스를 보여줍니다.

```python
code_artifact = wandb.Artifact(type="code")
code_artifact.add_file("./train.py")
wandb.log_artifact(code_artifact)
```
