---
title: What happens if I pass a class attribute into wandb.log()?
menu:
  support:
    identifier: ja-support-kb-articles-pass_class_attribute_wandblog
support:
- experiments
toc_hide: true
type: docs
url: /support/:filename
---

`wandb.log()` にクラス属性を渡すのは避けてください。属性はネットワーク呼び出しが実行される前に変更される可能性があります。 メトリクスをクラス属性として保存する場合は、ディープコピーを使用して、ログに記録されたメトリクスが `wandb.log()` 呼び出し時の属性の 値 と一致するようにしてください。
