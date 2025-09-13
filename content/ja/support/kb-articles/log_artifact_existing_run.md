---
title: 既存の run にアーティファクトをログするにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-log_artifact_existing_run
support:
- artifacts
toc_hide: true
type: docs
url: /support/:filename
---

場合によっては、以前にログした run の出力としてアーティファクトを指定する必要があります。この場合は、古い run を再初期化し、次のように新しいアーティファクトをログします:

```python
with wandb.init(id="existing_run_id", resume="allow") as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    artifact.add_file("my_data/file.txt")
    run.log_artifact(artifact)
```