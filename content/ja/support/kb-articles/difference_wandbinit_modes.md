---
title: wandb.init モードの違いは何ですか？
menu:
  support:
    identifier: ja-support-kb-articles-difference_wandbinit_modes
support:
- 実験管理
toc_hide: true
type: docs
url: /ja/support/:filename
---

これらのモードが利用可能です:

* `online` (デフォルト): クライアントはデータをwandb サーバーに送信します。
* `offline`: クライアントはデータをwandb サーバーに送信する代わりに、マシン上にローカルで保存します。後でデータを同期するには、[`wandb sync`]({{< relref path="/ref/cli/wandb-sync.md" lang="ja" >}}) コマンドを使用してください。
* `disabled`: クライアントはモックされたオブジェクトを返すことで操作をシミュレートし、ネットワーク通信を防ぎます。すべてのログはオフになりますが、すべてのAPI メソッドスタブは呼び出し可能なままです。このモードは通常、テストに使用されます。