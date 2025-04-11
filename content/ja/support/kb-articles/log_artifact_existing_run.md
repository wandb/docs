---
title: 既存の run に Artifacts をログするにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-log_artifact_existing_run
support:
- アーティファクト
toc_hide: true
type: docs
url: /support/:filename
---

時々、以前にログを記録した run の出力としてアーティファクトをマークする必要があります。この場合、古い run を再初期化し、新しいアーティファクトを次のようにログします:

```python
with wandb.init(id="existing_run_id", resume="allow") as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    artifact.add_file("my_data/file.txt")
    run.log_artifact(artifact)
```