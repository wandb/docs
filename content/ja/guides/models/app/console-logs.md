---
title: コンソールログ
menu:
  default:
    identifier: ja-guides-models-app-console-logs
url: guides/app/console-logs
---

実験を実行すると、コンソールにさまざまなメッセージが表示されることがあります。W&B はこれらのコンソールログをキャプチャして W&B App に表示します。これらのメッセージを利用して、実験のデバッグや進行状況の監視ができます。

## コンソールログの表示

W&B App で run のコンソールログにアクセスするには、以下の手順に従います。

1. W&B App 内で、あなたの Project に移動します。
2. **Runs** テーブルから任意の run を選択します。
3. Project のサイドバーで **Logs** タブをクリックします。

{{% alert %}}
ストレージ制限のため、ログの最大 10,000 行のみが表示されます
{{% /alert %}}


## コンソールログの種類

W&B は、情報メッセージ、警告、エラーなど、複数種類のコンソールログをキャプチャします。各ログの重要度はプレフィックスで区別できます。

### 情報メッセージ
情報メッセージは、run の進行状況や状態に関する情報を提供します。通常は `wandb:` が先頭につきます。

```text
wandb: Starting Run: abc123
wandb: Run data is saved locally in ./wandb/run-20240125_120000-abc123
```

### 警告メッセージ
実行を止めるほどではない、潜在的な問題に関する警告は `WARNING:` で始まります。

```text
WARNING Found .wandb file, not streaming tensorboard metrics.
WARNING These runs were logged with a previous version of wandb.
```

### エラーメッセージ 
重大な問題が発生した場合、`ERROR:` で始まるエラーメッセージが表示されます。これらは run が正常に完了できない場合に生じます。

```text
ERROR Unable to save notebook session history.
ERROR Failed to save notebook.
```

## コンソールログの設定

コード内で `wandb.Settings` オブジェクトを `wandb.init()` に渡すことで、W&B がコンソールログをどのように扱うかを設定できます。`wandb.Settings` では、以下のパラメータを指定してコンソールログの振る舞いを調整できます。

- `show_errors`: `True` にするとエラーメッセージが W&B App に表示され、`False` では表示されません。
- `silent`: `True` にすると W&B の全てのコンソール出力が抑制されます。 本番（プロダクション）環境など、コンソールのノイズを最小限にしたい場合に便利です。
- `show_warnings`: `True` で警告メッセージを W&B App に表示、`False` で表示しません。
- `show_info`: `True` で情報メッセージを W&B App に表示、`False` で表示しません。

以下は設定例です：

```python
import wandb

settings = wandb.Settings(
    show_errors=True,  # W&B App でエラーメッセージを表示する
    silent=False,      # すべての W&B コンソール出力を無効化
    show_warnings=True # W&B App で警告メッセージを表示する
)

with wandb.init(settings=settings) as run:
    # ここにトレーニングコードを記述
    run.log({"accuracy": 0.95})
```

## カスタムロギング

W&B はアプリケーションのコンソールログをキャプチャしますが、独自のロギング設定には干渉しません。Python の組み込み `print()` 関数や `logging` モジュールで自由にメッセージを記録できます。

```python
import wandb

with wandb.init(project="my-project") as run:
    for i in range(100, 1000, 100):
        # これは W&B にもログされ、コンソールにも出力される
        run.log({"epoch": i, "loss": 0.1 * i})
        print(f"epoch: {i} loss: {0.1 * i}")
```

コンソールログの出力例は以下のようになります：

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

各コンソールログには自動的にタイムスタンプが付与されます。これにより、各ログメッセージがいつ生成されたかの追跡が可能です。

コンソールログのタイムスタンプは表示・非表示を切り替えられます。コンソールページの左上にある **Timestamp visible** ドロップダウンから、タイムスタンプの表示/非表示を選択できます。

## コンソールログの検索

コンソールログページ上部の検索バーを使って、キーワードでログをフィルタリングできます。特定の語句やラベル、エラーメッセージを検索することができます。

## カスタムラベルによるフィルタ

{{% alert color="secondary"  %}}
`x_` で始まるパラメータ（たとえば `x_label`）はパブリックプレビュー中です。フィードバックがあれば [W&B リポジトリの GitHub issue](https://github.com/wandb/wandb) へご記入ください。
{{% /alert %}}

`wandb.Settings` の `x_label` 引数で指定したラベルを使い、コンソールログページ上部 UI の検索バーからラベルでログを絞り込むことができます。

```python
import wandb

# プライマリノードで run を初期化
run = wandb.init(
    entity="entity",
    project="project",
	settings=wandb.Settings(
        x_label="custom_label"  # （オプション）ログをフィルタするためのカスタムラベル
        )
)
```

## コンソールログのダウンロード

W&B App で run のコンソールログをダウンロードするには：

1. W&B App で Project に移動します。
2. **Runs** テーブルから run を選択します。
3. Project のサイドバーから **Logs** タブをクリックします。
4. コンソールログページ右上のダウンロードボタンをクリックします。


## コンソールログのコピー

W&B App で run のコンソールログをコピーするには：

1. W&B App で Project に移動します。
2. **Runs** テーブルから run を選択します。
3. Project のサイドバーから **Logs** タブをクリックします。
4. コンソールログページ右上のコピー（copy）ボタンをクリックします。