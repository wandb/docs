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

これらのモードが利用可能です:

* `online` (デフォルト): クライアントはデータを wandb サーバーに送信します。
* `offline`: クライアントはデータを wandb サーバーに送信する代わりに、ローカルのマシンにデータを保存します。データを後で同期するには、[`wandb sync`]({{< relref path="/ref/cli/wandb-sync.md" lang="ja" >}}) コマンドを使用してください。
* `disabled`: クライアントがオペレーションをシミュレートし、モックオブジェクトを返すことで、あらゆるネットワーク通信を防ぎます。すべてのログはオフになりますが、すべての API メソッドのスタブは呼び出せます。このモードは通常、テストに使用されます。