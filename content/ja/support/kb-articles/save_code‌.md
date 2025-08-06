---
title: コードを保存するにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-save_code‌
support:
- アーティファクト
toc_hide: true
type: docs
url: /support/:filename
---

`wandb.init` で `save_code=True` を使うと、その run を開始するメインのスクリプトやノートブックを保存できます。run のすべてのコードを保存したい場合は、コードを Artifacts でバージョン管理してください。以下の例はこのプロセスを示しています。

```python
# コード用のアーティファクトを作成
code_artifact = wandb.Artifact(type="code")
# コードファイルを追加
code_artifact.add_file("./train.py")
# アーティファクトをログ
wandb.log_artifact(code_artifact)
```