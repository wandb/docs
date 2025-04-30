---
title: ブール変数をハイパーパラメーターとしてフラグを立てることはできますか？
menu:
  support:
    identifier: ja-support-kb-articles-flag_boolean_variables_hyperparameters
support:
- sweep
toc_hide: true
type: docs
url: /support/:filename
---

設定のコマンドセクションでハイパーパラメーターをブールフラグとして渡すには、`${args_no_boolean_flags}` マクロを使用します。このマクロは、ブールパラメータを自動的にフラグとして含めます。もし `param` が `True` の場合、コマンドは `--param` を受け取ります。`param` が `False` の場合、フラグは省略されます。