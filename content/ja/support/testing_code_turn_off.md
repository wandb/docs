---
title: Can I turn off wandb when testing my code?
menu:
  support:
    identifier: ja-support-testing_code_turn_off
tags:
- artifacts
toc_hide: true
type: docs
---

`wandb.init(mode="disabled")` を使用するか、`WANDB_MODE=disabled` を設定して、W&Bをテスト目的で無操作 (NOOP) に設定します。

{{% alert %}}
`wandb.init(mode="disabled")` を使用しても、W&B がアーティファクトを `WANDB_CACHE_DIR` に保存するのを防ぐことはできません。
{{% /alert %}}