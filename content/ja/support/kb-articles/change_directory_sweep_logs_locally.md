---
title: How can I change the directory my sweep ログ to locally?
menu:
  support:
    identifier: ja-support-kb-articles-change_directory_sweep_logs_locally
support:
- sweeps
toc_hide: true
type: docs
url: /support/:filename
---

W&B run のデータのログディレクトリーを設定するには、環境変数 `WANDB_DIR` を設定します。例えば：

```python
os.environ["WANDB_DIR"] = os.path.abspath("your/directory")
```