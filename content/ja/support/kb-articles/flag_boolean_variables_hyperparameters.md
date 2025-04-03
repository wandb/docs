---
title: Can we flag boolean variables as hyperparameters?
menu:
  support:
    identifier: ja-support-kb-articles-flag_boolean_variables_hyperparameters
support:
- sweeps
toc_hide: true
type: docs
url: /support/:filename
---

設定のコマンドセクションで `${args_no_boolean_flags}` マクロを使用すると、ハイパーパラメーターをブール値フラグとして渡すことができます。このマクロは、ブール値のパラメータを自動的にフラグとして含めます。`param` が `True` の場合、コマンドは `--param` を受け取ります。`param` が `False` の場合、フラグは省略されます。
