---
title: ローカルで sweep の ログを保存するディレクトリーを変更するにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-change_directory_sweep_logs_locally
support:
- sweeps
toc_hide: true
type: docs
url: /support/:filename
---

環境変数 `WANDB_DIR` を使って、W&B の run データのログ用ディレクトリーを指定します。例:

```python
os.environ["WANDB_DIR"] = os.path.abspath("your/directory")
```