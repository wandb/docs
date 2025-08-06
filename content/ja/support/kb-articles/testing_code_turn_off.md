---
title: テスト時に wandb をオフにすることはできますか？
url: /support/:filename
toc_hide: true
type: docs
support:
- アーティファクト
---

`wandb.init(mode="disabled")` を使用するか、`WANDB_MODE=disabled` を設定することで、テスト目的で W&B をノーオペレーション（NOOP）として設定できます。

{{% alert %}}
`wandb.init(mode="disabled")` を使用しても、W&B は `WANDB_CACHE_DIR` にアーティファクトを保存することを防げません。
{{% /alert %}}