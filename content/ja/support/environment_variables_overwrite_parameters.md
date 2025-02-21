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

`wandb.init` に渡された引数は、環境変数を上書きします。環境変数が設定されていないときにシステムのデフォルト以外のディレクトリーをデフォルトとして設定するには、`wandb.init(dir=os.getenv("WANDB_DIR", my_default_override))` を使用してください。