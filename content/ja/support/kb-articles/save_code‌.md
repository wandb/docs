---
title: コードをどのように保存しますか？
menu:
  support:
    identifier: ja-support-kb-articles-save_code‌
support:
- artifacts
toc_hide: true
type: docs
url: /support/:filename
---

`wandb.init` で `save_code=True` を使用して、run を開始するメインの スクリプト または ノートブック を保存します。run の すべての コード を保存するには、Artifacts を使って コード を バージョン管理 してください。次の例はこの手順を示します:

```python
code_artifact = wandb.Artifact(type="code")
code_artifact.add_file("./train.py")
wandb.log_artifact(code_artifact)
```