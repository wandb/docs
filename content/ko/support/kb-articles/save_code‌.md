---
title: 코드를 어떻게 저장하나요?
menu:
  support:
    identifier: ko-support-kb-articles-save_code‌
support:
- 아티팩트
toc_hide: true
type: docs
url: /support/:filename
---

`wandb.init`에서 `save_code=True`를 사용하면 run을 시작하는 메인 스크립트나 노트북이 저장됩니다. run에 사용된 모든 코드를 저장하려면 Artifacts를 이용해 코드를 버전 관리하세요. 아래 예시는 그 과정을 보여줍니다:

```python
code_artifact = wandb.Artifact(type="code")
code_artifact.add_file("./train.py")  # 코드 파일을 추가합니다
wandb.log_artifact(code_artifact)     # 코드 아티팩트를 로깅합니다
```