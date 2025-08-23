---
title: テスト時に wandb をオフにすることはできますか？
menu:
  support:
    identifier: ja-support-kb-articles-testing_code_turn_off
support:
- アーティファクト
toc_hide: true
type: docs
url: /support/:filename
---

`wandb.init(mode="disabled")` を使用するか、`WANDB_MODE=disabled` を設定することで、テスト目的で W&B を何もしない（NOOP）モードに設定できます。

{{% alert %}}
`wandb.init(mode="disabled")` を使用しても、W&B がアーティファクトを `WANDB_CACHE_DIR` に保存することは防げません。
{{% /alert %}}