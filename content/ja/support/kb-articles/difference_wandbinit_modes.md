---
title: wandb.init モードの違いは何ですか？
menu:
  support:
    identifier: ja-support-kb-articles-difference_wandbinit_modes
support:
- experiments
toc_hide: true
type: docs
url: /support/:filename
---

これらのモードが利用可能です:

* `online` (デフォルト): クライアントは データ を wandb サーバー に送信します。
* `offline`: クライアントは データ を wandb サーバー に送信する代わりに、マシン上にローカルで保存します。後でデータを同期するには、[`wandb sync`]({{< relref path="/ref/cli/wandb-sync.md" lang="ja" >}}) コマンド を使用してください。
* `disabled`: クライアントはモックされたオブジェクトを返すことで操作をシミュレートし、ネットワーク通信を防ぎます。すべての ログ はオフになりますが、すべての API メソッド スタブは呼び出し可能なままです。このモードは通常、テストに使用されます。