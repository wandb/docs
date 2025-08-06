---
title: ブール変数をハイパーパラメーターとしてフラグ付けできますか？
url: /support/:filename
toc_hide: true
type: docs
support:
- スイープ
---

設定のコマンドセクションで `${args_no_boolean_flags}` マクロを使うと、ハイパーパラメーターをブール値のフラグとして渡すことができます。このマクロは、ブール値のパラメータを自動的にフラグとして含めます。`param` が `True` の場合、コマンドには `--param` が渡されます。`param` が `False` の場合、そのフラグは省略されます。