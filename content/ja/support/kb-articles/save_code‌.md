---
title: How do I save code?‌
menu:
  support:
    identifier: ja-support-kb-articles-save_code‌
support:
- artifacts
toc_hide: true
type: docs
url: /support/:filename
---

`wandb.init` で `save_code=True` を使用すると、run を起動するメイン スクリプトまたはノートブックが保存されます。run のすべてのコードを保存するには、Artifacts でコードをバージョン管理します。次の例で、このプロセスを説明します。

```python
code_artifact = wandb.Artifact(type="code")
code_artifact.add_file("./train.py")
wandb.log_artifact(code_artifact)
```
