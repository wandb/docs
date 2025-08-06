---
title: wandb.init のモードの違いは何ですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- 実験
---

これらのモードが利用できます:

* `online`（デフォルト）: クライアントがデータを wandb サーバーへ送信します。
* `offline`: クライアントがデータを wandb サーバーへ送信せず、ローカルマシンに保存します。後でデータを同期するには [`wandb sync`]({{< relref "/ref/cli/wandb-sync.md" >}}) コマンドを使用してください。
* `disabled`: クライアントが動作をシミュレートし、モックされたオブジェクトを返してネットワーク通信を一切行いません。すべてのログが無効化されますが、すべての API メソッドスタブは呼び出し可能なままです。主にテスト用途で使われます。