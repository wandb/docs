---
title: What is the difference between wandb.init modes?
menu:
  support:
    identifier: ja-support-kb-articles-difference_wandbinit_modes
support:
- experiments
toc_hide: true
type: docs
url: /support/:filename
---

利用可能なモードは以下のとおりです。

* `online` (デフォルト): クライアントは wandb サーバー に データ を送信します。
* `offline`: クライアントは データ を wandb サーバー に送信する代わりに、ローカルマシンに データ を保存します。後で データ を同期するには、[`wandb sync`]({{< relref path="/ref/cli/wandb-sync.md" lang="ja" >}}) コマンド を使用します。
* `disabled`: クライアントはモック オブジェクト を返すことによって操作をシミュレートし、ネットワーク通信を防止します。すべての ログ はオフになりますが、すべての API メソッド スタブ は呼び出し可能です。このモードは通常、テストに使用されます。
