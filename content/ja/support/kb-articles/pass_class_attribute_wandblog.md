---
title: wandb.Run.log() に クラス属性 を渡すとどうなりますか？
menu:
  support:
    identifier: ja-support-kb-articles-pass_class_attribute_wandblog
support:
- experiments
toc_hide: true
type: docs
url: /support/:filename
---

クラス属性を `wandb.Run.log()` に渡すのは避けてください。属性はネットワーク呼び出しが実行される前に変更される可能性があります。メトリクスをクラス属性として保持する場合は、`wandb.Run.log()` 呼び出し時点での属性の値とログされたメトリクスが一致するように、ディープコピーを使用してください。