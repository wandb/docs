---
title: 環境変数は wandb.init() に渡したパラメータを上書きしますか？
menu:
  support:
    identifier: ja-support-kb-articles-environment_variables_overwrite_parameters
support:
- 環境変数
toc_hide: true
type: docs
url: /support/:filename
---

`wandb.init` に渡された引数は環境変数を上書きします。環境変数が設定されていない場合に、システムのデフォルトではない既定のディレクトリーを設定するには、`wandb.init(dir=os.getenv("WANDB_DIR", my_default_override))` を使用します。