---
title: What is the difference between wandb.init modes?
menu:
  support:
    identifier: ja-support-difference_wandbinit_modes
tags:
- experiments
toc_hide: true
type: docs
---

利用可能なモードは次のとおりです。

*   `online` (デフォルト): クライアントはデータを wandb サーバー に送信します。
*   `offline`: クライアントはデータを wandb サーバー に送信せず、代わりにマシン上にローカルに保存します。後でデータを同期するには、[`wandb sync`]({{< relref path="/ref/cli/wandb-sync.md" lang="ja" >}}) コマンドを使用します。
*   `disabled`: クライアントは、モックされたオブジェクトを返すことによって操作をシミュレートし、ネットワーク通信を防ぎます。すべてのログ記録はオフになりますが、すべての API メソッド のスタブは呼び出し可能です。このモードは通常、テストに使用されます。
