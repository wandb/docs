---
displayed_sidebar: default
---


# コードの保存

デフォルトでは、最新のGitコミットハッシュのみを保存します。UIで実験間のコードを動的に比較できるようにするため、より多くのコード機能を有効にすることができます。

`wandb` バージョン 0.8.28 から、`wandb.init()` を呼び出すメインのトレーニングファイルからコードを保存できます。このコードはダッシュボードに同期され、runページのタブやCode Comparerパネルに表示されます。デフォルトでコード保存を有効にするには、[設定ページ](https://app.wandb.ai/settings)にアクセスしてください。

![Here's what your account settings look like. You can save code by default.](/images/app_ui/code_saving.png)

## ライブラリのコードを保存

コード保存が有効な場合、wandbは `wandb.init()` を呼び出したファイルからコードを保存します。追加のライブラリコードを保存するには、2つのオプションがあります：

* `wandb.init()` を呼び出した後に `wandb.run.log_code(".")` を呼び出します
* `wandb.init` に settings オブジェクトと code\_dirを指定して渡します： `wandb.init(settings=wandb.Settings(code_dir="."))`

これにより、現在のディレクトリーおよびすべてのサブディレクトリー内のすべてのPythonソースコードファイルが [artifact](../../../../ref/python/artifact.md) としてキャプチャされます。保存されるソースコードファイルのタイプや場所をさらに制御するには、[リファレンスドキュメント](../../../../ref/python/run.md#log_code)を参照してください。

## Code Comparer

ワークスペースやレポートで**+**ボタンをクリックして新しいパネルを追加し、Code Comparerを選択します。プロジェクト内の任意の二つのExperimentsを比較して、変更されたコード行を正確に確認できます。以下は例です：

![](/images/app_ui/code_comparer.png)

## Jupyterセッションの履歴

**wandb** バージョン 0.8.34 から、当社のライブラリはJupyterセッションの保存を行います。Jupyter内で **wandb.init()** を呼び出すと、現在のセッションで実行されたコードの履歴を含むJupyter notebookを自動的に保存するフックを追加します。このセッション履歴は run のファイルブラウザー内のコードディレクトリーに表示されます：

![](/images/app_ui/jupyter_session_history.png)

このファイルをクリックすると、セッションで実行されたセルとiPythonの `display` メソッドを呼び出すことで生成されたすべての出力が表示されます。これにより、特定のrun内でJupyterで実行されたコードを正確に確認できます。可能な場合、ノートブックの最新バージョンも保存され、それもコードディレクトリーに表示されます。

![](/images/app_ui/jupyter_session_history_display.png)

## Jupyterのdiff機能

もう一つのボーナス機能として、ノートブックのdiff機能があります。Code Comparerパネルで生のJSONを表示する代わりに、各セルを抽出し、変更された行を表示します。Jupyterをより深くプラットフォームに統合するためのエキサイティングな機能がいくつか計画されています。