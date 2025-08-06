---
title: 環境変数は wandb.init() に渡したパラメータを上書きしますか？
url: /support/:filename
toc_hide: true
type: docs
support:
- 環境変数
---

`wandb.init` に渡された引数は環境変数を上書きします。環境変数が設定されていない場合にシステムのデフォルト以外のディレクトリーをデフォルトに設定したい場合は、`wandb.init(dir=os.getenv("WANDB_DIR", my_default_override))` を使用してください。