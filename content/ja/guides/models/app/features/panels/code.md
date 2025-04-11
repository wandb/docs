---
title: コードを保存して差分を取る
menu:
  default:
    identifier: ja-guides-models-app-features-panels-code
    parent: panels
weight: 50
---

デフォルトでは、W&Bは最新のgitコミットハッシュのみを保存します。UIで実験間のコードを動的に比較するためのより多くのコード機能を有効にできます。

`wandb` バージョン 0.8.28 から、W&Bは `wandb.init()` を呼び出すメインのトレーニングファイルからコードを保存することができます。

## ライブラリコードを保存する

コード保存を有効にすると、W&Bは `wandb.init()` を呼び出したファイルからコードを保存します。追加のライブラリコードを保存するには、以下の3つのオプションがあります:

### `wandb.init` を呼び出した後に `wandb.run.log_code(".")` を呼び出す

```python
import wandb

wandb.init()
wandb.run.log_code(".")
```

### `code_dir` を設定して `wandb.init` に設定オブジェクトを渡す

```python
import wandb

wandb.init(settings=wandb.Settings(code_dir="."))
```

これにより、現在のディレクトリーおよびすべてのサブディレクトリー内のPythonソースコードファイルが[アーティファクト]({{< relref path="/ref/python/artifact.md" lang="ja" >}})としてキャプチャされます。保存されるソースコードファイルの種類と場所をより詳細に制御するには、[リファレンスドキュメント]({{< relref path="/ref/python/run.md#log_code" lang="ja" >}})を参照してください。

### UIでコード保存を設定する

コード保存をプログラム的に設定する以外に、W&Bアカウントの設定でこの機能を切り替えることもできます。これを有効にすると、アカウントに関連付けられているすべてのチームでコード保存が有効になります。

> デフォルトでは、W&Bはすべてのチームでコード保存を無効にします。

1. W&Bアカウントにログインします。
2. **設定** > **プライバシー** に移動します。
3. **プロジェクトとコンテンツのセキュリティ** の下で、**デフォルトのコード保存を無効にする** をオンにします。

## コードコンペアラー

異なるW&B runで使用されたコードを比較する:

1. ページの右上隅にある **パネルを追加** ボタンを選択します。
2. **TEXT AND CODE** ドロップダウンを展開し、**コード** を選択します。

{{< img src="/images/app_ui/code_comparer.png" alt="" >}}

## Jupyterセッション履歴

W&BはJupyterノートブックセッションで実行されたコードの履歴を保存します。Jupyter内で**wandb.init()** を呼び出すと、W&Bは現在のセッションで実行されたコードの履歴を含むJupyterノートブックを自動的に保存するフックを追加します。

1. コードが含まれているプロジェクトワークスペースに移動します。
2. 左ナビゲーションバーの**Artifacts** タブを選択します。
3. **コード**アーティファクトを展開します。
4. **ファイル**タブを選択します。

{{< img src="/images/app_ui/jupyter_session_history.gif" alt="" >}}

これは、セッションで実行されたセルと、iPythonの表示メソッドを呼び出して作成された出力を表示します。これにより、指定されたrunのJupyter内でどのコードが実行されたかを正確に確認することができます。可能な場合、W&Bはノートブックの最新バージョンも保存し、コードディレクトリー内で見つけることができます。

{{< img src="/images/app_ui/jupyter_session_history_display.png" alt="" >}}