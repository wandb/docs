---
title: How do I log an artifact to an existing run?
menu:
  support:
    identifier: ja-support-kb-articles-log_artifact_existing_run
support:
- artifacts
toc_hide: true
type: docs
url: /support/:filename
---

以前に記録された run の出力として Artifact をマークする必要がある場合があります。この場合、古い run を再度初期化し、次のように新しい Artifact をログに記録します。

```python
with wandb.init(id="existing_run_id", resume="allow") as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    artifact.add_file("my_data/file.txt")
    run.log_artifact(artifact)
```
