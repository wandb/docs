---
title: Do environment variables overwrite the parameters passed to wandb.init()?
menu:
  support:
    identifier: ja-support-environment_variables_overwrite_parameters
tags:
- environment variables
toc_hide: true
type: docs
---

`wandb.init` に渡された 引数 は、 環境 変数 を上書きします。 環境 変数 が設定されていない場合、システム デフォルト以外のデフォルト ディレクトリー を設定するには、 `wandb.init(dir=os.getenv("WANDB_DIR", my_default_override))` を使用します。
