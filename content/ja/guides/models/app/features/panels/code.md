---
title: Save and diff code
menu:
  default:
    identifier: ja-guides-models-app-features-panels-code
    parent: panels
weight: 50
---

デフォルトでは、W&B は最新の git コミットハッシュのみを保存します。より多くのコード機能を有効にすると、UI で実験間のコードを動的に比較できます。

`wandb` バージョン 0.8.28 以降、W&B は `wandb.init()` を呼び出すメインのトレーニングファイルからコードを保存できます。

## ライブラリコードを保存する

コードの保存を有効にすると、W&B は `wandb.init()` を呼び出したファイルからコードを保存します。追加のライブラリコードを保存するには、次の 3 つのオプションがあります。

### `wandb.init()` を呼び出した後、`wandb.run.log_code(".")` を呼び出す

```python
import wandb

wandb.init()
wandb.run.log_code(".")
```

### `code_dir` が設定された settings オブジェクトを `wandb.init` に渡す

```python
import wandb

wandb.init(settings=wandb.Settings(code_dir="."))
```

これにより、現在のディレクトリーおよびすべてのサブディレクトリーにあるすべての Python ソースコードファイルが [アーティファクト]({{< relref path="/ref/python/artifact.md" lang="ja" >}}) としてキャプチャされます。保存されるソースコードファイルのタイプと場所をより詳細に制御するには、[リファレンスドキュメント]({{< relref path="/ref/python/run.md#log_code" lang="ja" >}}) を参照してください。

### UI でコードの保存を設定する

プログラムでコードの保存を設定するだけでなく、W&B アカウントの Settings でこの機能を切り替えることもできます。これにより、アカウントに関連付けられているすべての Teams に対してコードの保存が有効になることに注意してください。

> デフォルトでは、W&B はすべての Teams に対してコードの保存を無効にします。

1. W&B アカウントにログインします。
2. **Settings** > **Privacy** に移動します。
3. **Project and content security** で、**Disable default code saving** をオンにします。

## コード比較ツール
異なる W&B の Runs で使用されるコードを比較します。

1. ページの右上隅にある **Add panels** ボタンを選択します。
2. **TEXT AND CODE** ドロップダウンを展開し、**Code** を選択します。

{{< img src="/images/app_ui/code_comparer.png" alt="" >}}

## Jupyter セッション履歴

W&B は、Jupyter notebook セッションで実行されたコードの履歴を保存します。Jupyter 内で **wandb.init()** を呼び出すと、W&B は現在のセッションで実行されたコードの履歴を含む Jupyter notebook を自動的に保存する フック を追加します。

1. コードが含まれている プロジェクト の ワークスペース に移動します。
2. 左側の ナビゲーションバー で **Artifacts** タブを選択します。
3. **code** アーティファクト を展開します。
4. **Files** タブを選択します。

{{< img src="/images/app_ui/jupyter_session_history.gif" alt="" >}}

これにより、セッションで実行されたセルと、iPython の display メソッドを呼び出すことによって作成された出力が表示されます。これにより、特定の run 内の Jupyter で実行されたコードを正確に確認できます。可能な場合、W&B はコード ディレクトリー にもある最新バージョンの notebook も保存します。

{{< img src="/images/app_ui/jupyter_session_history_display.png" alt="" >}}
