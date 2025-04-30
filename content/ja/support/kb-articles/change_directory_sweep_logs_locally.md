---
title: 自分の sweep ログを保存するディレクトリーをローカルに変更するにはどうすればいいですか？
menu:
  support:
    identifier: ja-support-kb-articles-change_directory_sweep_logs_locally
support:
- sweep
toc_hide: true
type: docs
url: /ja/support/:filename
---

W&B run のデータのログディレクトリーを設定するには、環境変数 `WANDB_DIR` を設定します。例えば：

```python
os.environ["WANDB_DIR"] = os.path.abspath("your/directory")
```