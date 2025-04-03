---
title: 'How can I fix an error like `AttributeError: module ''wandb'' has no attribute
  ...`?'
menu:
  support:
    identifier: ja-support-kb-articles-how_can_i_resolve_the_attributeerror_module_wandb_has_no_attribute
support:
- crashing and hanging runs
toc_hide: true
type: docs
url: /support/:filename
---

Python で `wandb` をインポートする際に、`AttributeError: module 'wandb' has no attribute 'init'` や `AttributeError: module 'wandb' has no attribute 'login'` のようなエラーが発生した場合、`wandb` がインストールされていないか、インストールが破損している可能性があります。ただし、現在の作業ディレクトリーには `wandb` ディレクトリーが存在します。このエラーを修正するには、`wandb` をアンインストールし、そのディレクトリーを削除してから、`wandb` をインストールします。

```bash
pip uninstall wandb; rm -rI wandb; pip install wandb
```
