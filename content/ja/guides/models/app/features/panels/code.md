---
title: コードを保存して差分を比較
menu:
  default:
    identifier: ja-guides-models-app-features-panels-code
    parent: panels
weight: 50
---

デフォルトでは、W&B は最新の git コミットハッシュのみを保存します。UI で実験間のコードを動的に比較できる、より多くのコード関連機能を有効にできます。

`wandb` バージョン 0.8.28 以降、`wandb.init()` を呼び出したメインのトレーニング用ファイルのコードを W&B が保存できるようになりました。

## ライブラリのコードを保存する

コード保存を有効にすると、W&B は `wandb.init()` を呼び出したファイルのコードを保存します。追加のライブラリコードを保存するには、次の 3 つの方法があります。

### `wandb.init()` の後に `wandb.Run.log_code(".")` を呼び出す

```python
import wandb

with wandb.init() as run:
  run.log_code(".")
```

### `code_dir` を設定した settings オブジェクトを `wandb.init()` に渡す

```python
import wandb

wandb.init(settings=wandb.Settings(code_dir="."))
```

これにより、現在のディレクトリーとそのすべてのサブディレクトリー内にある Python のソースコードファイルが、[artifact]({{< relref path="/ref/python/sdk/classes/artifact.md" lang="ja" >}}) として保存されます。保存するソースコードファイルの種類や場所をより詳細に制御したい場合は、[reference docs]({{< relref path="/ref/python/sdk/classes/run.md#log_code" lang="ja" >}}) を参照してください。

### UI でコード保存を設定する

プログラムからコード保存を設定する以外に、W&B アカウントの **Settings** でもこの機能を切り替えられます。これを有効にすると、あなたのアカウントに関連付けられたすべての Teams でコード保存が有効になります。

> デフォルトでは、W&B はすべての Teams でコード保存を無効化しています。

1. W&B アカウントにログインします。
2. **Settings** > **Privacy** に移動します。
3. **Project and content security** の下にある **Disable default code saving** をオンに切り替えます。

## コード比較
異なる W&B Runs で使用したコードを比較します。

1. ページ右上の **Add panels** ボタンを選択します。
2. **TEXT AND CODE** ドロップダウンを開き、**Code** を選択します。

{{< img src="/images/app_ui/code_comparer.png" alt="コード比較パネル" >}}

## Jupyter セッション履歴

W&B は Jupyter ノートブックセッションで実行したコードの履歴を保存します。Jupyter の中で **wandb.init()** を呼び出すと、W&B はフックを追加し、現在のセッションで実行したコードの履歴を含む Jupyter ノートブックを自動的に保存します。

1. コードを含むプロジェクトの Workspace に移動します。
2. 左のナビゲーションバーで **Artifacts** タブを選択します。
3. **code** Artifact を展開します。
4. **Files** タブを選択します。

{{< img src="/images/app_ui/jupyter_session_history.gif" alt="Jupyter セッション履歴" >}}

これにより、セッションで実行されたセルと、IPython の display メソッドを呼び出して作成された出力が表示されます。これによって、特定の run で Jupyter 内で実行されたコードを正確に確認できます。可能な場合、W&B はノートブックの最新バージョンも保存し、code ディレクトリー内でも見つけられます。

{{< img src="/images/app_ui/jupyter_session_history_display.png" alt="Jupyter セッションの出力" >}}