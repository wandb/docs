---
title: wandb.log() にクラス属性を渡した場合どうなりますか？
menu:
  support:
    identifier: ja-support-kb-articles-pass_class_attribute_wandblog
support:
- experiments
toc_hide: true
type: docs
url: /support/:filename
---

クラス属性を `wandb.log()` に渡すことは避けてください。属性はネットワーク呼び出しが実行される前に変更される可能性があります。メトリクスをクラス属性として保存する場合は、ログに記録されたメトリクスが `wandb.log()` 呼び出し時の属性の値と一致するように、ディープコピーを使用してください。