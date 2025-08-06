---
title: コンソールログ
url: guides/app/console-logs
---

実験を実行すると、さまざまなメッセージがコンソールに表示されることがあります。W&B はコンソールログをキャプチャし、W&B App で表示します。これらのメッセージを使って、実験のデバッグや進捗の監視ができます。

## コンソールログの表示

W&B App で run のコンソールログに アクセス する方法:

1. W&B App で自分の project に移動します。
2. **Runs** テーブルから run を選択します。
3. project サイドバーの **Logs** タブをクリックします。

{{% alert %}}
ストレージ制限のため、ログは最大 10,000 行まで表示されます
{{% /alert %}}


## コンソールログの種類

W&B はコンソールから情報メッセージ、警告、エラーなど、いくつかの種類のログをキャプチャします。ログの重大度を示すプレフィックスが付与されます。

### 情報メッセージ
情報メッセージは run の進捗や状態をお知らせします。通常、`wandb:` で始まります。

```text
wandb: Starting Run: abc123
wandb: Run data is saved locally in ./wandb/run-20240125_120000-abc123
```

### 警告メッセージ
実行を停止しない注意事項には `WARNING:` が付きます。

```text
WARNING Found .wandb file, not streaming tensorboard metrics.
WARNING These runs were logged with a previous version of wandb.
```

### エラーメッセージ
深刻な問題が発生した場合は `ERROR:` で始まります。これらは run の正常終了を妨げる可能性があります。

```text
ERROR Unable to save notebook session history.
ERROR Failed to save notebook.
```

## コンソールログの設定

コード内で、`wandb.init()` に `wandb.Settings` オブジェクトを渡すことで、W&B がコンソールログをどのように扱うかを 設定 できます。`wandb.Settings` では、以下のパラメータでコンソールログの振る舞いを制御できます。

- `show_errors`: `True` にするとエラーメッセージが W&B App に表示されます。`False` だと表示されません。
- `silent`: `True` にすると、すべての W&B コンソール出力が抑制されます。コンソールのノイズを最小限にしたいプロダクション環境で便利です。
- `show_warnings`: `True` なら警告メッセージを W&B App で表示します。`False` なら非表示です。
- `show_info`: `True` なら情報メッセージを W&B App で表示します。`False` なら非表示です。

次の例では、これらの 設定 の使い方を示します。

```python
import wandb

settings = wandb.Settings(
    show_errors=True,  # W&B App にエラーメッセージを表示
    silent=False,      # W&B のすべてのコンソール出力を無効化
    show_warnings=True # W&B App に警告メッセージを表示
)

with wandb.init(settings=settings) as run:
    # ここにトレーニングコードを記述
    run.log({"accuracy": 0.95})
```

## カスタムロギング

W&B はアプリケーションからのコンソールログもキャプチャしますが、独自のロギング設定を邪魔することはありません。Python 標準の `print()` 関数や `logging` モジュールを自由に利用できます。

```python
import wandb

with wandb.init(project="my-project") as run:
    for i in range(100, 1000, 100):
        # これは W&B にもログされ、コンソールにも出力されます
        run.log({"epoch": i, "loss": 0.1 * i})
        print(f"epoch: {i} loss: {0.1 * i}")
```

コンソールログは以下のように表示されます:

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

コンソールログの各エントリにはタイムスタンプが自動で付加されます。これにより、それぞれのログメッセージがいつ生成されたかを追跡できます。

タイムスタンプの表示・非表示は切り替え可能です。コンソールページの左上にある **Timestamp visible** のドロップダウンから、タイムスタンプを表示するかどうかを選択できます。

## コンソールログ検索

コンソールログページ上部の検索バーで、キーワードによるログのフィルタができます。特定の単語、ラベル、エラーメッセージなどを検索可能です。

## カスタムラベルでのフィルター

{{% alert color="secondary"  %}}
`x_` プレフィックス（例：`x_label`）が付いたパラメータはパブリックプレビュー中です。フィードバックがある場合には [W&B リポジトリの GitHub issue](https://github.com/wandb/wandb) へご投稿ください。
{{% /alert %}}

`wandb.Settings` の `x_label` 引数で ラベル を渡すことで、コンソールログページ上部の検索バーからそのラベルでログのフィルタが行えます。

```python
import wandb

# プライマリノードで run を初期化
run = wandb.init(
    entity="entity",
    project="project",
	settings=wandb.Settings(
        x_label="custom_label"  # （オプション）ログフィルタ用カスタムラベル
        )
)
```

## コンソールログのダウンロード

W&B App で run のコンソールログをダウンロードするには:

1. W&B App で自分の project に移動します。
2. **Runs** テーブルから run を選択します。
3. project サイドバーの **Logs** タブをクリックします。
4. コンソールログページ右上のダウンロードボタンをクリックします。

## コンソールログのコピー

W&B App で run のコンソールログをコピーするには:

1. W&B App で自分の project に移動します。
2. **Runs** テーブルから run を選択します。
3. project サイドバーの **Logs** タブをクリックします。
4. コンソールログページ右上のコピーアイコンをクリックします。