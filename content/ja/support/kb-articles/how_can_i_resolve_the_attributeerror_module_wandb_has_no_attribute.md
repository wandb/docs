---
title: '`AttributeError: module ''wandb'' has no attribute ...` のようなエラーはどうすれば解決できますか？'
url: /support/:filename
toc_hide: true
type: docs
support:
- クラッシュやハングアップする run
---

Python で `wandb` をインポートした際に `AttributeError: module 'wandb' has no attribute 'init'` や `AttributeError: module 'wandb' has no attribute 'login'` というエラーが発生する場合、`wandb` がインストールされていないか、インストールが壊れていることに加え、現在の作業ディレクトリーに `wandb` ディレクトリーが存在している可能性があります。このエラーを解決するには、`wandb` をアンインストールし、ディレクトリーを削除してから再インストールしてください。

```bash
pip uninstall wandb; rm -rI wandb; pip install wandb
```