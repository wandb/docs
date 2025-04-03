---
title: Can I turn off wandb when testing my code?
menu:
  support:
    identifier: ja-support-kb-articles-testing_code_turn_off
support:
- artifacts
toc_hide: true
type: docs
url: /support/:filename
---

テスト目的で W&B を操作なし (NOOP) として設定するには、`wandb.init(mode="disabled")` を使用するか、`WANDB_MODE=disabled` を設定します。

{{% alert %}}
`wandb.init(mode="disabled")` を使用しても、W&B が Artifacts を `WANDB_CACHE_DIR` に保存することを防ぐことはできません。
{{% /alert %}}
