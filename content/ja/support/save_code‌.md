---
title: How do I save code?‌
menu:
  support:
    identifier: ja-support-save_code‌
tags:
- artifacts
toc_hide: true
type: docs
---

`wandb.init` で `save_code=True` を使用して、run を起動するメインのスクリプトまたはノートブックを保存します。run のすべてのコードを保存するには、Artifacts でコードをバージョン管理します。次の例はこのプロセスを示しています。

```python
code_artifact = wandb.Artifact(type="code")  # コードのアーティファクトを作成
code_artifact.add_file("./train.py")  # ファイルを追加
wandb.log_artifact(code_artifact)  # アーティファクトをログ
```