---
title: wandb.init のモードの違いは何ですか？
menu:
  support:
    identifier: ja-support-kb-articles-difference_wandbinit_modes
support:
- 実験
toc_hide: true
type: docs
url: /support/:filename
---

以下のモードが利用できます：

* `online`（デフォルト）：クライアントが wandb サーバーにデータを送信します。
* `offline`：クライアントは wandb サーバーにデータを送信せず、データをローカルマシンに保存します。後でデータを同期するには [`wandb sync`]({{< relref path="/ref/cli/wandb-sync.md" lang="ja" >}}) コマンドを使用してください。
* `disabled`：クライアントは動作をシミュレートし、モックされたオブジェクトを返してネットワーク通信を抑止します。すべてのログは無効になりますが、全ての API メソッドのスタブは呼び出し可能なままです。このモードは主にテスト用途で使用されます。