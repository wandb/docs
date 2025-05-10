---
title: コードをテストするときに wandb をオフにできますか？
menu:
  support:
    identifier: ja-support-kb-articles-testing_code_turn_off
support:
  - artifacts
toc_hide: true
type: docs
url: /ja/support/:filename
translationKey: testing_code_turn_off
---
`wandb.init(mode="disabled")` を使用するか、`WANDB_MODE=disabled` を設定して、W&B をテスト目的で何もしない (NOOP) 状態に設定します。

{{% alert %}}
`wandb.init(mode="disabled")` を使用しても、W&B が Artifacts を `WANDB_CACHE_DIR` に保存するのを防ぐことはできません。
{{% /alert %}}