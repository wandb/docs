---
title: How can I change the directory my sweep logs to locally?
menu:
  support:
    identifier: ja-support-kb-articles-change_directory_sweep_logs_locally
support:
- sweeps
toc_hide: true
type: docs
url: /support/:filename
---

`WANDB_DIR` 環境変数を構成することで、W&B の run データ の ログ ディレクトリー を設定します。例：

```python
os.environ["WANDB_DIR"] = os.path.abspath("your/directory")
```
