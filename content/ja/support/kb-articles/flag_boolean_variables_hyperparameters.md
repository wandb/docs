---
title: ブール変数をハイパーパラメーターとしてフラグ付けできますか？
menu:
  support:
    identifier: ja-support-kb-articles-flag_boolean_variables_hyperparameters
support:
- sweeps
toc_hide: true
type: docs
url: /support/:filename
---

ハイパーパラメーターをブール フラグとして渡すには、設定のコマンド セクションで `${args_no_boolean_flags}` マクロを使用します。このマクロは、ブール パラメータをフラグとして自動的に付与します。`param` が `True` の場合、コマンドは `--param` を受け取ります。`param` が `False` の場合、フラグは省略されます。