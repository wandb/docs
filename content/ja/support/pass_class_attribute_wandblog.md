---
title: What happens if I pass a class attribute into wandb.log()?
menu:
  support:
    identifier: ja-support-pass_class_attribute_wandblog
tags:
- experiments
toc_hide: true
type: docs
---

`wandb.log()` にクラス属性を渡すのは避けてください。属性は、ネットワーク呼び出しが実行される前に変更される可能性があります。 メトリクス をクラス属性として保存する場合は、ディープコピーを使用して、 ログ に記録された メトリクス が `wandb.log()` の呼び出し時の属性の 値 と一致するようにしてください。
