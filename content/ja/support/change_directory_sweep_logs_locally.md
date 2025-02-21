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

`WANDB_DIR` 環境変数を設定して、W&B の run データの ログディレクトリー を設定します。例：

```python
os.environ["WANDB_DIR"] = os.path.abspath("your/directory") # W&B の ログデータ を保存するディレクトリー の絶対パスを設定します
```
