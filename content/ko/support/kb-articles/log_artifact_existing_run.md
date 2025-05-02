---
title: How do I log an artifact to an existing run?
menu:
  support:
    identifier: ko-support-kb-articles-log_artifact_existing_run
support:
- artifacts
toc_hide: true
type: docs
url: /ko/support/:filename
translationKey: log_artifact_existing_run
---
이따금씩, 이전에 기록된 run의 결과로 아티팩트를 표시해야 할 때가 있습니다. 이 경우, 이전 run을 다시 초기화하고 다음과 같이 새로운 아티팩트를 기록합니다.

```python
with wandb.init(id="existing_run_id", resume="allow") as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    artifact.add_file("my_data/file.txt")
    run.log_artifact(artifact)
```
