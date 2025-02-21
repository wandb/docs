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

Python で `wandb` をインポートする際に、`AttributeError: module 'wandb' has no attribute 'init'` や `AttributeError: module 'wandb' has no attribute 'login'` のようなエラーが発生した場合、`wandb` がインストールされていないか、インストールが破損しているものの、現在の作業ディレクトリー に `wandb` ディレクトリー が存在していることが考えられます。このエラーを修正するには、`wandb` をアンインストールし、ディレクトリー を削除してから、`wandb` をインストールしてください。

```bash
pip uninstall wandb; rm -rI wandb; pip install wandb
```
