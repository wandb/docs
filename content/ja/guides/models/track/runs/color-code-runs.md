---
title: セマンティック run プロット凡例
description: チャートのためのセマンティック凡例を作成する
menu:
  default:
    identifier: color-code-runs
    parent: what-are-runs
---

W&B の run をメトリクスや設定パラメータで色分けすることで、視覚的に意味のある折れ線グラフや凡例を作成できます。パフォーマンスの高低や最新値など、実験ごとのメトリクスによって run を色分けすることで、パターンやトレンドを簡単に特定できます。W&B は、選択したパラメータの値に基づいて run を自動的に色分けバケットにグループ化します。

ワークスペースの設定ページから、run のメトリクスや設定ベースのカラーを設定できます。

1. W&B の Project に移動します。
2. Project サイドバーから **Workspace** タブを選択します。
3. 右上の **Settings** アイコン（⚙️）をクリックします。
4. ドロワーから **Runs** を選択し、**Key-based colors** を選択します。
    - **Key** ドロップダウンから、run の色分けに使いたいメトリクスを選択します。
    - **Y value** ドロップダウンから、run の色分けに使いたい y 値（指標）を選択します。
    - バケット（区切り）の数を 2〜8 の範囲で設定します。

以下のセクションでは、メトリクスや y 値の設定方法、run の色分けに使うバケットのカスタマイズ方法を解説します。

## メトリクスを設定する

**Key** ドロップダウンのメトリクスは、[W&B にログした key-value ペア]({{< relref "guides/models/track/runs/color-code-runs/#custom-metrics" >}}) や、W&B で定義されている [デフォルトメトリクス]({{< relref "guides/models/track/runs/color-code-runs/#default-metrics" >}}) から選ばれます。

### デフォルトメトリクス

* `Relative Time (Process)`: run 開始時点からの経過秒数（プロセス時間）。
* `Relative Time (Wall)`: run 開始時点からの経過秒数（ウォールクロック時間ベース）。
* `Wall Time`: エポック（1970年1月1日）からの秒数（ウォールクロック時間）。
* `Step`: run のステップ数で、トレーニングや評価の進捗追跡によく使われます。

### カスタムメトリクス

トレーニングや評価スクリプトでログしたカスタムメトリクスに基づき run を色付け、わかりやすいプロット凡例を作成できます。カスタムメトリクスは key-value ペアとしてログし、key がメトリクス名、value が値です。

例として、以下のコードスニペットは、トレーニングループ中に精度（`"acc"`）と損失（`"loss"`）をログします。

```python
import wandb
import random

epochs = 10

with wandb.init(project="basic-intro") as run:
  # このブロックはメトリクスを記録するトレーニングループを模倣しています
  offset = random.random() / 5
  for epoch in range(2, epochs):
      acc = 1 - 2 ** -epoch - random.random() / epoch - offset
      loss = 2 ** -epoch + random.random() / epoch + offset

      # スクリプトから W&B へメトリクスをログします
      run.log({"acc": acc, "loss": loss})
```

**Key** ドロップダウンには `"acc"` と `"loss"` のどちらも選択肢として表示されます。

## 設定キーを設定する

**Key** ドロップダウンの設定項目は、W&B run の初期化時に `config` パラメータとして渡した key-value ペアから選ばれます。設定キーは主にハイパーパラメーターや、トレーニング・評価スクリプトに使うその他の設定値のログに使われます。

```python
import wandb

config = {
  "learning_rate": 0.01,
  "batch_size": 32,
  "optimizer": "adam"
}

with wandb.init(project="basic-intro", config=config) as run:
  # ここにトレーニングのコードを書きます
  pass
```

**Key** ドロップダウンには `"learning_rate"`、`"batch_size"`、`"optimizer"` が表示されます。

## y 値を設定する

以下のオプションから選択できます。

- **Latest**: 各線の最後に記録された Y 値で色分けします。
- **Max**: メトリクスの中で記録された最高値で色分けします。
- **Min**: メトリクスの中で記録された最小値で色分けします。

## バケットをカスタマイズする

バケットは、選択したメトリクスまたは設定キーの値の範囲に基づいて W&B が run を分類するための区切りです。バケットは指定したメトリクスや設定キーの値の範囲で均等に分割され、それぞれにユニークな色が割り当てられます。run がそのバケットの範囲内に入ると、該当する色で表示されます。

例えば、以下のようになります。

{{< img src="/images/track/color-coding-runs.png" alt="Color coded runs" >}}

- **Key** が `"Accuracy"`（省略形 `"acc"`）に設定されています。
- **Y value** が `"Max"` に設定されています。

この設定では、W&B はそれぞれの run の精度値に基づいて色分けを行います。色は薄い黄色から濃い色までグラデーションになっています。薄い色は低い精度、濃い色は高い精度を表します。

このメトリクス用に 6 つのバケットが定義され、それぞれが精度値の範囲を表します。**Buckets** セクションには以下の範囲が設定されています。

- Bucket 1: (Min - 0.7629)
- Bucket 2: (0.7629 - 0.7824)
- Bucket 3: (0.7824 - 0.8019)
- Bucket 4: (0.8019 - 0.8214)
- Bucket 5: (0.8214 - 0.8409)
- Bucket 6: (0.8409 - Max)

下記の折れ線グラフでは、最も高い精度（0.8232）の run が濃い紫（Bucket 5）で表示され、最も低い精度（0.7684）の run は薄いオレンジ（Bucket 2）で表示されています。他の run もそれぞれの精度値に合わせて色分けされ、カラ―グラデーションによって相対的なパフォーマンスが分かるようになっています。

{{< img src="/images/track/color-code-runs-plot.png" alt="Color coded runs plot" >}}