---
title: コンソール ログ
menu:
  default:
    identifier: ja-guides-models-app-console-logs
url: guides/app/console-logs
---

実験を実行すると、コンソールにさまざまなメッセージが表示されることに気づくかもしれません。W&B はコンソールログを取得し、W&B App に表示します。これらのメッセージを使用して、実験の振る舞いをデバッグおよび監視できます。

## コンソールログを表示

W&B App で run のコンソールログにアクセスするには:

1. W&B App で自分の Project に移動します。
2. **Runs** テーブル内の run を選択します。
3. Project サイドバーの **Logs** タブをクリックします。

{{% alert %}}
ストレージの制限により、ログは最大 10,000 行までしか表示されません
{{% /alert %}}




## コンソールログの種類

W&B は複数種類のコンソールログを取得します。情報メッセージ、警告、エラーで、重大度を示すプレフィックスが付きます。

### 情報メッセージ
情報メッセージは、run の進行状況やステータスに関する更新を提供します。通常は `wandb:` が先頭に付きます。

```text
wandb: Starting Run: abc123
wandb: Run data is saved locally in ./wandb/run-20240125_120000-abc123
```

### 警告メッセージ
実行を停止しない潜在的な問題に関する警告には `WARNING:` が付きます。

```text
WARNING Found .wandb file, not streaming tensorboard metrics.
WARNING These runs were logged with a previous version of wandb.
```

### エラーメッセージ 
重大な問題のエラーメッセージには `ERROR:` が付きます。run が正常に完了しない可能性のある問題を示します。

```text
ERROR Unable to save notebook session history.
ERROR Failed to save notebook.
```

## コンソールログの設定

コード内で `wandb.Settings` オブジェクトを `wandb.init()` に渡し、W&B がコンソールログをどのように扱うかを設定します。`wandb.Settings` では、コンソールログの振る舞いを制御するために次のパラメータを設定できます。

- `show_errors`: `True` の場合、エラーメッセージが W&B App に表示されます。`False` の場合、エラーメッセージは表示されません。
- `silent`: `True` の場合、すべての W&B コンソール出力を抑制します。コンソールのノイズを最小化したい プロダクション 環境で便利です。
- `show_warnings`: `True` の場合、警告メッセージが W&B App に表示されます。`False` の場合、警告メッセージは表示されません。
- `show_info`: `True` の場合、情報メッセージが W&B App に表示されます。`False` の場合、情報メッセージは表示されません。

以下はこれらの設定方法の例です。

```python
import wandb

settings = wandb.Settings(
    show_errors=True,  # W&B App にエラーメッセージを表示
    silent=False,      # すべての W&B コンソール出力を無効化
    show_warnings=True # W&B App に警告メッセージを表示
)

with wandb.init(settings=settings) as run:
    # ここにトレーニング コードを書きます
    run.log({"accuracy": 0.95})
```

## カスタム ログ記録

W&B は アプリケーション のコンソールログを取り込みますが、独自のロギング設定には干渉しません。Python の組み込みの `print()` 関数や `logging` モジュールでメッセージをログできます。

```python
import wandb

with wandb.init(project="my-project") as run:
    for i in range(100, 1000, 100):
        # これは W&B にログされ、コンソールにも出力されます
        run.log({"epoch": i, "loss": 0.1 * i})
        print(f"epoch: {i} loss: {0.1 * i}")
```

コンソールログは次のようになります。

```text
1 epoch:  100 loss: 1.3191105127334595
2 epoch:  200 loss: 0.8664389848709106
3 epoch:  300 loss: 0.6157898902893066
4 epoch:  400 loss: 0.4961796700954437
5 epoch:  500 loss: 0.42592573165893555
6 epoch:  600 loss: 0.3771176040172577
7 epoch:  700 loss: 0.3393910825252533
8 epoch:  800 loss: 0.3082585036754608
9 epoch:  900 loss: 0.28154927492141724
```

## タイムスタンプ

各コンソールログのエントリには自動的にタイムスタンプが追加されます。これにより、各ログメッセージがいつ生成されたかを追跡できます。

コンソールログのタイムスタンプはオン／オフを切り替えできます。コンソールページ左上の **Timestamp visible** ドロップダウンで、表示するか非表示にするかを選択してください。

## コンソールログを検索

コンソールログページ上部の検索バーで、キーワードに基づいてログをフィルタリングできます。特定の用語、ラベル、エラーメッセージを検索できます。

## カスタム ラベルでフィルタリング

{{% alert color="secondary"  %}}
`x_` で始まるパラメータ（`x_label` など）はパブリックプレビュー中です。フィードバックは [GitHub issue を W&B リポジトリに作成](https://github.com/wandb/wandb) してください。
{{% /alert %}}

コンソールログページ上部の UI 検索バーで、`wandb.Settings` の `x_label` に引数として渡したラベルに基づいてコンソールログをフィルタリングできます。 

```python
import wandb

# プライマリノードで run を初期化
run = wandb.init(
    entity="entity",
    project="project",
	settings=wandb.Settings(
        x_label="custom_label"  # （任意）ログをフィルタリングするためのカスタムラベル
        )
)
```

## コンソールログをダウンロード

W&B App で run のコンソールログをダウンロードするには:

1. W&B App で自分の Project に移動します。
2. **Runs** テーブル内の run を選択します。
3. Project サイドバーの **Logs** タブをクリックします。
4. コンソールログページ右上のダウンロードボタンをクリックします。


## コンソールログをコピー

W&B App で run のコンソールログをコピーするには:

1. W&B App で自分の Project に移動します。
2. **Runs** テーブル内の run を選択します。
3. Project サイドバーの **Logs** タブをクリックします。
4. コンソールログページ右上のコピー ボタンをクリックします。