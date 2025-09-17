---
title: セマンティックな run プロットの凡例
description: チャート用のセマンティックな凡例を作成する
menu:
  default:
    identifier: ja-guides-models-track-runs-color-code-runs
    parent: what-are-runs
---

メトリクスや設定パラメータに基づいて W&B の Runs を色分けし、意味のある折れ線プロットと凡例を作成できます。パフォーマンスのメトリクス（最大、最小、最新の値）で Runs を色分けすることで、experiments 全体のパターンやトレンドを見つけやすくなります。W&B は選択したパラメータの値に基づき、Runs を色分けされたバケットに自動でグループ化します。

Workspace の設定ページに移動して、Runs の色をメトリクスまたは設定ベースで設定します:

1. W&B の project に移動します。
2. project のサイドバーから **Workspace** タブを選択します。
3. 右上の **Settings** アイコン (⚙️) をクリックします。
4. ドロワーで **Runs** を選び、続いて **Key-based colors** を選択します。
    - **Key** ドロップダウンから、Runs の色分けに使うメトリクスを選びます。
    - **Y value** ドロップダウンから、Runs の色分けに使う y 値を選びます。
    - バケットの数を 2 から 8 の間で設定します。

以下では、メトリクスと y 値の設定方法、そして Runs の色分けに使うバケットのカスタマイズ方法を説明します。

## メトリクスを設定する

**Key** ドロップダウンに表示されるメトリクスの候補は、[W&B にログした内容]({{< relref path="guides/models/track/runs/color-code-runs/#custom-metrics" lang="ja" >}}) と、W&B が定義する [既定のメトリクス]({{< relref path="guides/models/track/runs/color-code-runs/#default-metrics" lang="ja" >}}) に基づきます。

### 既定のメトリクス

* `Relative Time (Process)`: run の相対時間。run の開始からの経過秒数。
* `Relative Time (Wall)`: run の相対時間。run の開始からの経過秒数（ウォールクロック時間で補正）。
* `Wall Time`: run のウォールクロック時刻。エポックからの秒数。
* `Step`: run のステップ番号。通常はトレーニングや評価の進捗トラッキングに用います。

### カスタムメトリクス

トレーニングや評価のスクリプトがログするカスタムメトリクスに基づいて Runs を色分けし、意味のある凡例を作成できます。カスタムメトリクスはキーと値のペアとしてログされ、キーはメトリクス名、値はそのメトリクスの値です。

例えば、次のコードスニペットはトレーニングループ中に精度（"acc" キー）と損失（"loss" キー）をログします:

```python
import wandb
import random

epochs = 10

with wandb.init(project="basic-intro") as run:
  # メトリクスをログするトレーニングループを模したブロック
  offset = random.random() / 5
  for epoch in range(2, epochs):
      acc = 1 - 2 ** -epoch - random.random() / epoch - offset
      loss = 2 ** -epoch + random.random() / epoch + offset

      # スクリプトから W&B にメトリクスをログする
      run.log({"acc": acc, "loss": loss})
```

**Key** ドロップダウンには、"acc" と "loss" の両方が選択肢として表示されます。

## 設定キーを選ぶ

**Key** ドロップダウンに表示される設定の候補は、W&B の run を初期化するときに `config` パラメータへ渡したキーと値のペアから取得されます。設定キーは、ハイパーパラメーターやトレーニング/評価スクリプトで使用するその他の設定をログするために使われます。

```python
import wandb

config = {
  "learning_rate": 0.01,
  "batch_size": 32,
  "optimizer": "adam"
}

with wandb.init(project="basic-intro", config=config) as run:
  # ここにトレーニングのコードを書く
  pass
```

**Key** ドロップダウンには、"learning_rate"、"batch_size"、"optimizer" が選択肢として表示されます。

## y 値を設定する

次のいずれかを選べます:

- **Latest**: 各線の最後にログされたステップ時点の Y 値で色を決めます。
- **Max**: そのメトリクスでログされた最大の Y 値で色を決めます。
- **Min**: そのメトリクスでログされた最小の Y 値で色を決めます。

## バケットをカスタマイズする

バケットは、選択したメトリクスまたは設定キーに基づいて Runs を分類するために W&B が用いる値の範囲です。指定したメトリクスまたは設定キーの値の範囲全体を均等に分割し、各バケットに固有の色が割り当てられます。各バケットの範囲に入る Runs は、その色で表示されます。

次の例を考えます:

{{< img src="/images/track/color-coding-runs.png" alt="色分けされた Runs" >}}

- **Key** は "Accuracy"（略して "acc"）に設定されています。
- **Y value** は "Max" に設定されています。

この設定では、W&B は各 run をその精度の値に基づいて色分けします。色は淡い黄色から濃い色までグラデーションになります。色が淡いほど精度が低く、濃いほど精度が高いことを表します。

このメトリクスには 6 個のバケットが定義されており、それぞれが精度の値の範囲を表します。**Buckets** セクションには、次のバケット範囲が定義されています:

- バケット 1: (Min - 0.7629)
- バケット 2: (0.7629 - 0.7824)
- バケット 3: (0.7824 - 0.8019)
- バケット 4: (0.8019 - 0.8214)
- バケット 5: (0.8214 - 0.8409)
- バケット 6: (0.8409 - Max)

下の折れ線プロットでは、最も精度が高い run（0.8232）は濃い紫（バケット 5）、最も精度が低い run（0.7684）は淡いオレンジ（バケット 2）で表示されています。その他の Runs も精度の値に基づいて色分けされ、色のグラデーションが相対的なパフォーマンスを示します。

{{< img src="/images/track/color-code-runs-plot.png" alt="色分けされた Runs のプロット" >}}