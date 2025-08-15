---
title: sweep のログをローカルで保存するディレクトリーを変更するにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-change_directory_sweep_logs_locally
support:
- スイープ
toc_hide: true
type: docs
url: /support/:filename
---

W&B の run データのログディレクトリーは、環境変数 `WANDB_DIR` を設定することで指定できます。例えば:

```python
# ログディレクトリーを設定
os.environ["WANDB_DIR"] = os.path.abspath("your/directory")
```