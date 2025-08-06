---
title: wandb.Run.log() にクラス属性を渡した場合、どうなりますか？
url: /support/:filename
toc_hide: true
type: docs
support:
- 実験
---

クラス属性を `wandb.Run.log()` に渡すのは避けてください。属性はネットワークコールの実行前に変更される可能性があります。メトリクスをクラス属性として保存する場合は、deep copy を使用して、`wandb.Run.log()` 呼び出し時の属性の値とログされるメトリクスが一致するようにしてください。