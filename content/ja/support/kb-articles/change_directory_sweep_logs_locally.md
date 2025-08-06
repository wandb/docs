---
title: sweep のログをローカルで保存するディレクトリーを変更するにはどうすればいいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- スイープ
---

W&B run のデータを保存するログディレクトリーを設定するには、環境変数 `WANDB_DIR` を設定します。例えば、以下のように指定します。

```python
os.environ["WANDB_DIR"] = os.path.abspath("your/directory")
```