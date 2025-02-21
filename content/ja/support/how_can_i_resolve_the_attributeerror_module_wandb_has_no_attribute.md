---
title: 'How can I fix an error like `AttributeError: module ''wandb'' has no attribute
  ...`?'
menu:
  support:
    identifier: ja-support-how_can_i_resolve_the_attributeerror_module_wandb_has_no_attribute
tags:
- crashing and hanging runs
toc_hide: true
type: docs
---

`wandb` を Python でインポートする際に、`AttributeError: module 'wandb' has no attribute 'init'` や `AttributeError: module 'wandb' has no attribute 'login'` のようなエラーが発生した場合、`wandb` がインストールされていないか、インストールが破損している可能性がありますが、現在の作業ディレクトリーには `wandb` ディレクトリーが存在している可能性があります。このエラーを修正するには、`wandb` をアンインストールして、そのディレクトリーを削除し、再度 `wandb` をインストールしてください:

```bash
pip uninstall wandb; rm -rI wandb; pip install wandb
```