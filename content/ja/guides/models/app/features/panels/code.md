---
title: Save and diff code
menu:
  default:
    identifier: ja-guides-models-app-features-panels-code
    parent: panels
weight: 50
---

デフォルトで、W&B は最新の git コミットハッシュのみを保存します。UI 内で実験間のコードを動的に比較するためのコード機能をさらにオンにすることができます。

`wandb` バージョン 0.8.28 から、W&B は `wandb.init()` を呼び出すメイントレーニングファイルからのコードを保存できます。

## ライブラリコードを保存

コード保存を有効にすると、W&B は `wandb.init()` を呼び出したファイルからのコードを保存します。追加のライブラリコードを保存するには、以下の3つのオプションがあります。

### `wandb.init()` を呼んだ後に `wandb.run.log_code(".")` を呼び出す

```python
import wandb

wandb.init()
wandb.run.log_code(".")
```

### `code_dir` を設定して `settings` オブジェクトを `wandb.init` に渡す

```python
import wandb

wandb.init(settings=wandb.Settings(code_dir="."))
```

これにより、カレントディレクトリーとすべてのサブディレクトリーにあるすべての Python ソースコードファイルが [artifact]({{< relref path="/ref/python/artifact.md" lang="ja" >}}) としてキャプチャされます。保存されるソースコードファイルの種類や場所をより詳細に制御したい場合は、[リファレンスドキュメント]({{< relref path="/ref/python/run.md#log_code" lang="ja" >}}) をご覧ください。

### UIでコード保存を設定

プログラムによりコード保存を設定することに加え、W&B アカウントの設定でこの機能をトグルすることもできます。これにより、アカウントに関連付けられたすべてのチームでコード保存が有効になります。

> デフォルトでは、W&B はすべてのチームでコード保存を無効にしています。

1. W&B アカウントにログインします。
2. **Settings** > **Privacy** に移動します。
3. **Project and content security** の下で、**Disable default code saving** をオンにします。

## コードコンペアラー

異なる W&B runs に使用されたコードを比較：

1. ページの右上隅にある **Add panels** ボタンを選択します。
2. **TEXT AND CODE** ドロップダウンを展開し、**Code** を選択します。

{{< img src="/images/app_ui/code_comparer.png" alt="" >}}

## Jupyter セッション履歴

W&B は、Jupyter ノートブックセッションで実行されたコードの履歴を保存します。Jupyter 内で **wandb.init()** を呼び出すと、W&B はフックを追加して、現在のセッションで実行されたコードの履歴を含む Jupyter ノートブックを自動的に保存します。

1. コードを含むプロジェクトワークスペースに移動します。
2. 左ナビゲーションバーで **Artifacts** タブを選択します。
3. **code** artifact を展開します。
4. **Files** タブを選択します。

{{< img src="/images/app_ui/jupyter_session_history.gif" alt="" >}}

これにより、セッションで実行されたセルと iPython のディスプレイメソッドを呼び出すことで作成された出力が表示されます。これにより、Jupyter の特定の run 内でどのコードが実行されたか正確に確認できます。可能であれば、W&B はコードディレクトリーにあるノートブックの最新バージョンも保存します。

{{< img src="/images/app_ui/jupyter_session_history_display.png" alt="" >}}