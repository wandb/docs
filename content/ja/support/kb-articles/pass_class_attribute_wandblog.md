---
title: wandb.log() に クラス属性 を渡した場合どうなりますか？
menu:
  support:
    identifier: ja-support-kb-articles-pass_class_attribute_wandblog
support:
- 実験管理
toc_hide: true
type: docs
url: /ja/support/:filename
---

クラス属性を `wandb.log()` に渡すことは避けてください。属性はネットワーク呼び出しが実行される前に変更される可能性があります。メトリクスをクラス属性として保存する場合は、 ログ に記録されたメトリクスが `wandb.log()` 呼び出し時の属性の値と一致するように、ディープコピーを使用してください。