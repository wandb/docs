---
title: wandb.init のモードの違いは何ですか？
menu:
  support:
    identifier: ja-support-kb-articles-difference_wandbinit_modes
support:
- 実験管理
toc_hide: true
type: docs
url: /support/:filename
---

利用可能なモードは次のとおりです:

* `online`（デフォルト）: クライアントは wandb サーバーにデータを送信します。
* `offline`: クライアントは wandb サーバーに送信する代わりに、マシン上にデータをローカル保存します。後でデータを同期するには、[`wandb sync`]({{< relref path="/ref/cli/wandb-sync.md" lang="ja" >}}) コマンドを使用します。
* `disabled`: クライアントはモックのオブジェクトを返すことで動作をシミュレートし、あらゆるネットワーク通信を行わないようにします。すべてのログはオフになりますが、すべての API メソッドのスタブは引き続き呼び出し可能です。このモードは主にテストに使用されます。