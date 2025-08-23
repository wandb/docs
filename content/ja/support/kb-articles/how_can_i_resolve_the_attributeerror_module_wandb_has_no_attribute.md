---
title: '`AttributeError: module ''wandb'' has no attribute ...` のようなエラーをどうやって解決できますか？'
menu:
  support:
    identifier: ja-support-kb-articles-how_can_i_resolve_the_attributeerror_module_wandb_has_no_attribute
support:
- クラッシュやハングアップする run
toc_hide: true
type: docs
url: /support/:filename
---

Python で `wandb` をインポートした際に `AttributeError: module 'wandb' has no attribute 'init'` や `AttributeError: module 'wandb' has no attribute 'login'` のようなエラーが発生する場合、`wandb` がインストールされていないか、インストールが壊れている可能性がありますが、現在の作業ディレクトリーに `wandb` ディレクトリーが存在しています。このエラーを解決するには、`wandb` をアンインストールし、そのディレクトリーを削除してから `wandb` を再インストールしてください:

```bash
pip uninstall wandb; rm -rI wandb; pip install wandb
```