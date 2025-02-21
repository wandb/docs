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

`wandb.log()` にクラス属性を渡すのは避けましょう。属性はネットワークコールが実行される前に変更される可能性があります。メトリクスをクラス属性として保存する場合、ディープコピーを使用して、`wandb.log()` コール時の属性の値がログされたメトリクスと一致するようにしてください。