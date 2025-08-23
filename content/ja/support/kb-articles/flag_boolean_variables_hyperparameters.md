---
title: ブール変数をハイパーパラメーターとしてフラグ付けできますか？
menu:
  support:
    identifier: ja-support-kb-articles-flag_boolean_variables_hyperparameters
support:
- スイープ
toc_hide: true
type: docs
url: /support/:filename
---

コマンドの設定セクションで `${args_no_boolean_flags}` マクロを使用すると、ハイパーパラメーターをブール型フラグとして渡すことができます。このマクロは、ブール型のパラメータを自動的にフラグとして追加します。`param` が `True` の場合、コマンドには `--param` が渡されます。`param` が `False` の場合は、そのフラグは省略されます。