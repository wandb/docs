---
title: '`AttributeError: module ''wandb'' has no attribute ...` のようなエラーをどのように修正できますか？'
menu:
  support:
    identifier: ja-support-kb-articles-how_can_i_resolve_the_attributeerror_module_wandb_has_no_attribute
support:
- run のクラッシュとハングアップ
toc_hide: true
type: docs
url: /ja/support/:filename
---

`wandb` を Python でインポートする際に `AttributeError: module 'wandb' has no attribute 'init'` や `AttributeError: module 'wandb' has no attribute 'login'` といったエラーが発生した場合、`wandb` がインストールされていないか、インストールが破損している可能性がありますが、カレントワーキングディレクトリーには `wandb` ディレクトリーが存在しています。このエラーを修正するには、`wandb` をアンインストールし、そのディレクトリーを削除してから `wandb` をインストールしてください:

```bash
pip uninstall wandb; rm -rI wandb; pip install wandb
```