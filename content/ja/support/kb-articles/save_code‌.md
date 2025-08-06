---
title: コードを保存するにはどうしたらいいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- アーティファクト
---

`wandb.init` の `save_code=True` を使うと、その run を開始したメインのスクリプトやノートブックを保存できます。run のすべてのコードを保存したい場合は、Artifacts でコードをバージョン管理しましょう。以下の例はこのプロセスを示しています。

```python
code_artifact = wandb.Artifact(type="code")
code_artifact.add_file("./train.py")
wandb.log_artifact(code_artifact)
```