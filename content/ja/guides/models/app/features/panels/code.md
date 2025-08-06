---
title: コードを保存して差分を比較
menu:
  default:
    identifier: ja-guides-models-app-features-panels-code
    parent: panels
weight: 50
---

デフォルトでは、W&B は最新の git コミットハッシュのみを保存します。より多くのコード機能を有効にすると、UI 上で実験間のコード比較が動的にできるようになります。

`wandb` バージョン 0.8.28 以降、`wandb.init()` を呼び出すメイントレーニングファイルのコードを W&B が保存できるようになりました。

## ライブラリコードの保存

コード保存を有効にすると、W&B は `wandb.init()` を呼び出したファイルのコードを保存します。追加でライブラリのコードも保存したい場合、以下の 3 つの方法があります。

### `wandb.init()` の後に `wandb.Run.log_code(".")` を呼び出す

```python
import wandb

with wandb.init() as run:
  run.log_code(".")
```

### `code_dir` を指定した settings オブジェクトを `wandb.init()` に渡す

```python
import wandb

wandb.init(settings=wandb.Settings(code_dir="."))
```

これにより、現在のディレクトリーおよびすべてのサブディレクトリー内の全ての Python ソースコードファイルが [artifact]({{< relref path="/ref/python/sdk/classes/artifact.md" lang="ja" >}}) として保存されます。保存対象とするソースコードファイルの種類や場所をより細かく制御したい場合は、[リファレンスドキュメント]({{< relref path="/ref/python/sdk/classes/run.md#log_code" lang="ja" >}}) をご覧ください。

### UI でコード保存を設定する

プログラムで設定するだけでなく、W&B のアカウント Settings からもこの機能を切り替えることができます。この設定を有効にすると、アカウントに紐づくすべてのチームでコード保存が有効になります。

> デフォルトでは、W&B はすべてのチームでコード保存を無効にしています。

1. W&B アカウントにログインします。
2. **Settings** > **Privacy** に移動します。
3. **Project and content security** の項目で **Disable default code saving** をオンに切り替えます。

## コード比較ツール
異なる W&B run で使われたコードを比較できます。

1. ページ右上の **Add panels** ボタンを選択します。
2. **TEXT AND CODE** のドロップダウンを展開し、**Code** を選択します。

{{< img src="/images/app_ui/code_comparer.png" alt="Code comparer panel" >}}

## Jupyter セッション履歴

W&B は、Jupyter notebook セッションで実行されたコードの履歴を保存します。Jupyter 内で **wandb.init()** を呼び出すと、W&B はフックを追加し、そのセッションで実行したコード履歴を含む Jupyter notebook を自動的に保存します。

1. コードを含むプロジェクトの Workspace に移動します。
2. 左側のナビゲーションバーから **Artifacts** タブを選択します。
3. **code** artifact を展開します。
4. **Files** タブを選択します。

{{< img src="/images/app_ui/jupyter_session_history.gif" alt="Jupyter session history" >}}

これにより、セッション内で実行したセルと、iPython の display メソッドによって生成された出力が表示されます。これによって、特定の run で Jupyter 上で実際に実行したコードを正確に確認できます。可能な場合は、最新バージョンのノートブックも code ディレクトリーに保存されます。

{{< img src="/images/app_ui/jupyter_session_history_display.png" alt="Jupyter session output" >}}