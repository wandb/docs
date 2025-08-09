---
title: wandb.Run.log() にクラス属性を渡すとどうなりますか？
menu:
  support:
    identifier: ja-support-kb-articles-pass_class_attribute_wandblog
support:
- 実験
toc_hide: true
type: docs
url: /support/:filename
---

`wandb.Run.log()` にクラス属性を渡すのは避けてください。属性の内容はネットワーク呼び出し実行前に変更される可能性があります。メトリクスをクラス属性として保存している場合は、deep copy を使用して、`wandb.Run.log()` 呼び出し時点での属性の値がログされるようにしてください。