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

`wandb.init` で `save_code=True` を使用すると、run を起動するメインスクリプトまたは notebook が保存されます。run のすべてのコードを保存するには、Artifacts でコードをバージョン管理します。次の例は、このプロセスを示しています。

```python
code_artifact = wandb.Artifact(type="code")
code_artifact.add_file("./train.py")
wandb.log_artifact(code_artifact)
```
