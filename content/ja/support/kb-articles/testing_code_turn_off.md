---
title: コードをテストするときに wandb を無効化できますか？
menu:
  support:
    identifier: ja-support-kb-articles-testing_code_turn_off
support:
- artifacts
toc_hide: true
type: docs
url: /support/:filename
---

`wandb.init(mode="disabled")` を使うか、 `WANDB_MODE=disabled` を設定することで、テスト目的で W&B を no-operation（NOOP）として設定できます。

{{% alert %}}
`wandb.init(mode="disabled")` を使っても、W&B が `WANDB_CACHE_DIR` に artifacts を保存することは防げません。
{{% /alert %}}