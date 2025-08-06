---
title: 기존 run 에 아티팩트를 어떻게 로그하나요?
menu:
  support:
    identifier: ko-support-kb-articles-log_artifact_existing_run
support:
- 아티팩트
toc_hide: true
type: docs
url: /support/:filename
---

가끔 이전에 로그한 run의 출력으로 아티팩트를 표시해야 할 때가 있습니다. 이 경우, 이전 run을 다시 초기화하고 새로운 아티팩트를 아래와 같이 로그하세요:

```python
with wandb.init(id="existing_run_id", resume="allow") as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # 파일을 아티팩트에 추가합니다
    artifact.add_file("my_data/file.txt")
    # 아티팩트를 run에 로그합니다
    run.log_artifact(artifact)
```