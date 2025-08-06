---
title: セマンティック run プロット凡例
description: チャートにセマンティックな凡例を作成する
menu:
  default:
    identifier: ja-guides-models-track-runs-color-code-runs
    parent: what-are-runs
---

W&B の run をメトリクスや設定パラメータで色分けすることで、視覚的に意味を持たせた折れ線グラフや凡例を作成できます。パフォーマンス指標（最大値・最小値・最新値など）に基づいて run の色を分けることで、実験全体の傾向やパターンが一目で把握できます。W&B は選択したパラメータの値に基づいて、自動的に色分けされたバケットに run をグループ化します。

run をメトリクスまたは設定項目ごとに色分けするためには、ワークスペースの設定ページから以下の手順で設定してください。

1. W&B のプロジェクトにアクセスします。
2. プロジェクトサイドバーから **Workspace** タブを選択します。
3. 画面右上の **Settings** アイコン（⚙️）をクリックします。
4. ドロワーから **Runs** を選び、続いて **Key-based colors** を選択します。
    - **Key** ドロップダウンから、run に色を割り当てたいメトリクスを選択します。
    - **Y value** ドロップダウンから、色分けの基準となる y 値を選択します。
    - バケット数を 2 から 8 の範囲で指定します。

以下のセクションでは、メトリクスや y 値の設定方法と、run の色分けに使うバケットのカスタマイズ方法について説明します。

## メトリクスの設定

**Key** ドロップダウン内のメトリクス選択肢は、[W&B にログした key-value ペア]({{< relref path="guides/models/track/runs/color-code-runs/#custom-metrics" lang="ja" >}}) や W&B によって定義された [デフォルトメトリクス]({{< relref path="guides/models/track/runs/color-code-runs/#default-metrics" lang="ja" >}}) に基づいて自動生成されます。

### デフォルトメトリクス

* `Relative Time (Process)`: run の相対時間（run 開始からの秒数）を示します。
* `Relative Time (Wall)`: run の相対時間（run 開始からの秒数／ウォールクロック基準）を示します。
* `Wall Time`: run のウォールクロック時刻（エポックからの秒数）を示します。
* `Step`: run の step 数。通常、トレーニングや評価の進捗確認に使われます。

### カスタムメトリクス

トレーニングや評価スクリプトでログしたカスタムメトリクスを使って run を色分けし、意味のある凡例を作成できます。カスタムメトリクスは key-value ペアとしてログされ、key がメトリクス名、value がその値となります。

例えば、以下のコードスニペットでは、トレーニングループ内で accuracy（`"acc"` key）と loss（`"loss"` key）をログしています。

```python
import wandb
import random

epochs = 10

with wandb.init(project="basic-intro") as run:
  # このブロックは、トレーニングループでメトリクスをログする例です
  offset = random.random() / 5
  for epoch in range(2, epochs):
      acc = 1 - 2 ** -epoch - random.random() / epoch - offset
      loss = 2 ** -epoch + random.random() / epoch + offset

      # スクリプトから W&B へメトリクスをログ
      run.log({"acc": acc, "loss": loss})
```

**Key** ドロップダウンには `"acc"` と `"loss"` の両方が選択肢として表示されます。

## 設定キーの指定

**Key** ドロップダウンの設定項目は、W&B run を初期化する際に `config` パラメータへ渡した key-value ペアから生成されます。設定キーは、主にハイパーパラメーターやトレーニング・評価スクリプトの他の設定値のログに使われます。

```python
import wandb

config = {
  "learning_rate": 0.01,
  "batch_size": 32,
  "optimizer": "adam"
}

with wandb.init(project="basic-intro", config=config) as run:
  # ここにトレーニングのコードを記述
  pass
```

**Key** ドロップダウンには `"learning_rate"`、`"batch_size"`、`"optimizer"` が選択肢として利用できます。

## y 値の設定

以下の選択肢から選ぶことができます。

- **Latest**: 各線の最新 step での Y 値を基準に色分けします。
- **Max**: 指定したメトリクスでログされた最大の Y 値で色分けします。
- **Min**: 指定したメトリクスでログされた最小の Y 値で色分けします。

## バケットのカスタマイズ

バケットとは、選択したメトリクスや設定キーに基づいて W&B が run を分類する範囲（レンジ）のことです。バケットは指定したメトリクスや設定キーの値域を等分しており、それぞれに固有の色が割り当てられます。そのバケット内に該当する run は、対応する色で表示されます。

下記をご覧ください。

{{< img src="/images/track/color-coding-runs.png" alt="Color coded runs" >}}

- **Key** が `"Accuracy"`（略 `"acc"`）に設定されています。
- **Y value** が `"Max"` になっています。

この設定では、W&B は各 run の accuracy 値に基づいて色分けします。色合いは薄い黄色から濃い色まで変化します。薄い色は低い accuracy 値、濃い色は高い accuracy 値を示します。

この例では 6 つのバケットが定義されており、各バケットは accuracy 値の一定範囲を表します。**Buckets** セクションにて、バケットごとの範囲が以下のように定義されています。

- バケット 1: (Min - 0.7629)
- バケット 2: (0.7629 - 0.7824)
- バケット 3: (0.7824 - 0.8019)
- バケット 4: (0.8019 - 0.8214)
- バケット 5: (0.8214 - 0.8409)
- バケット 6: (0.8409 - Max)

下の折れ線グラフでは、最も高い accuracy（0.8232）の run は濃い紫色（バケット 5）で表示され、最も低い accuracy（0.7684）の run は薄いオレンジ色（バケット 2）で表されています。他の run も accuracy 値に応じて色が変化し、色のグラデーションでパフォーマンスの違いが視覚的に示されています。

{{< img src="/images/track/color-code-runs-plot.png" alt="Color coded runs plot" >}}