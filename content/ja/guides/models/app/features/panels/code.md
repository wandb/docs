---
title: コードを保存して差分を確認
menu:
  default:
    identifier: code
    parent: panels
weight: 50
---

デフォルトでは、W&B は最新の Git コミットハッシュのみを保存します。コード比較機能を有効にすると、UI 上で実験ごとのコードの違いを動的に比較できます。

`wandb` バージョン 0.8.28 以降、W&B は `wandb.init()` を呼び出したメイントレーニングファイルのコードを保存できます。

## ライブラリコードを保存する

コード保存を有効にすると、W&B は `wandb.init()` を呼び出したファイルのコードを保存します。追加のライブラリコードも保存したい場合は、次の3つの方法があります。

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

これにより、現在のディレクトリーおよびすべてのサブディレクトリー内のすべての Python ソースコードファイルが [artifact]({{< relref "/ref/python/sdk/classes/artifact.md" >}}) として記録されます。保存対象のソースコードファイルの種類や場所を細かく制御したい場合は、[リファレンスドキュメント]({{< relref "/ref/python/sdk/classes/run.md#log_code" >}}) をご覧ください。

### UI からコード保存を設定する

プログラムからだけでなく、ご自身の W&B アカウントの Settings 画面でも、この機能を有効・無効に切り替え可能です。なお、有効にした場合はアカウントに紐づくすべてのチームでコード保存が有効になります。

> デフォルトでは、すべてのチームでコード保存は無効化されています。

1. W&B アカウントにログインします。
2. **Settings** > **Privacy** に移動します。
3. **Project and content security** セクションで **Disable default code saving** をオンにします。

## コード比較機能

異なる W&B Run で使用したコードを比較できます。

1. 画面右上の **Add panels** ボタンを選択します。
2. **TEXT AND CODE** ドロップダウンを展開し、**Code** を選択します。

{{< img src="/images/app_ui/code_comparer.png" alt="Code comparer panel" >}}

## Jupyter セッション履歴

W&B は Jupyter ノートブックセッション内で実行したコードの履歴も保存します。Jupyter 内で **wandb.init()** を呼び出すと、W&B がフックを追加し、現在のセッションで実行されたコードの履歴が記録された Jupyter ノートブックを自動で保存します。

1. コードが含まれる Project の Workspace に移動します。
2. 左側のナビゲーションバーから **Artifacts** タブを選択します。
3. **code** アーティファクトを展開します。
4. **Files** タブを選択します。

{{< img src="/images/app_ui/jupyter_session_history.gif" alt="Jupyter session history" >}}

これにより、そのセッションで実行されたセルと iPython の display メソッドで生成された出力が表示されます。これによって、特定の Run で Jupyter 上でどのコードが実行されたかを正確に確認できます。可能な場合は、ノートブックの最新版もコードディレクトリー内に保存されます。

{{< img src="/images/app_ui/jupyter_session_history_display.png" alt="Jupyter session output" >}}