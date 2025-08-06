---
title: 既存の run にアーティファクトをログするにはどうすればいいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- アーティファクト
---

時々、以前にログ済みの run の出力としてアーティファクトをマークする必要がある場合があります。その場合は、以下のように古い run を再初期化し、新しいアーティファクトをログしてください。

```python
with wandb.init(id="existing_run_id", resume="allow") as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    # ファイルをアーティファクトに追加
    artifact.add_file("my_data/file.txt")
    # アーティファクトをログする
    run.log_artifact(artifact)
```
