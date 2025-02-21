---
title: Can we flag boolean variables as hyperparameters?
menu:
  support:
    identifier: ja-support-flag_boolean_variables_hyperparameters
tags:
- sweeps
toc_hide: true
type: docs
---

`${args_no_boolean_flags}` マクロを 設定 の コマンド セクションで使用すると、 ハイパーパラメータ をブール値フラグとして渡すことができます。このマクロは、ブール値 パラメータ を自動的にフラグとして含めます。`param` が `True` の場合、コマンドは `--param` を受け取ります。`param` が `False` の場合、フラグは省略されます。
