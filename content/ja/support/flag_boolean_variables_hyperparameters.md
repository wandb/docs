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

設定のコマンドセクションで `${args_no_boolean_flags}` マクロを使用して、ハイパーパラメーターをブールフラグとして渡します。このマクロは自動的にブールパラメータをフラグとして含めます。`param` が `True` の場合、コマンドは `--param` を受け取ります。`param` が `False` の場合、フラグは省略されます。