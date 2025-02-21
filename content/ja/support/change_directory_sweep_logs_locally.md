---
title: How can I change the directory my sweep logs to locally?
menu:
  support:
    identifier: ja-support-change_directory_sweep_logs_locally
tags:
- sweeps
toc_hide: true
type: docs
---

W&B run データのログ ディレクトリーを設定するには、環境変数 `WANDB_DIR` を構成します。例えば:

```python
os.environ["WANDB_DIR"] = os.path.abspath("your/directory")
```