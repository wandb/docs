---
title: オブジェクトの埋め込み
description: W&B の Embedding Projector では、ユーザーが PCA、UMAP、t-SNE など一般的な次元削減アルゴリズムを使って、多次元の埋め込みを
  2D 平面上にプロットできます。
menu:
  default:
    identifier: embedding-projector
    parent: query-panels
---

{{< img src="/images/weave/embedding_projector.png" alt="Embedding projector" >}}

[Embeddings](https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture) は、オブジェクト（人、画像、投稿、単語など）を一連の数字、つまり _ベクトル_ で表現するために使用されます。機械学習やデータサイエンスのユースケースでは、さまざまな手法や用途で Embeddings を生成することができます。このページでは、Embeddings についての基本的な知識がある読者が W&B 内で可視的に分析する方法に興味があることを前提としています。

## Embedding の例

- [ライブ インタラクティブ デモ Reports](https://wandb.ai/timssweeney/toy_datasets/reports/Feature-Report-W-B-Embeddings-Projector--VmlldzoxMjg2MjY4?accessToken=bo36zrgl0gref1th5nj59nrft9rc4r71s53zr2qvqlz68jwn8d8yyjdz73cqfyhq)
- [Colab サンプル](https://colab.research.google.com/drive/1DaKL4lZVh3ETyYEM1oJ46ffjpGs8glXA#scrollTo=D--9i6-gXBm_)

### Hello World

W&B では `wandb.Table` クラスを使って Embeddings をログすることができます。次の例は、5 次元からなる 3 つの Embedding をログするものです。

```python
import wandb

with wandb.init(project="embedding_tutorial") as run:
  embeddings = [
      # D1   D2   D3   D4   D5
      [0.2, 0.4, 0.1, 0.7, 0.5],  # embedding 1
      [0.3, 0.1, 0.9, 0.2, 0.7],  # embedding 2
      [0.4, 0.5, 0.2, 0.2, 0.1],  # embedding 3
  ]
  run.log(
      {"embeddings": wandb.Table(columns=["D1", "D2", "D3", "D4", "D5"], data=embeddings)}
  )
  run.finish()
```

上記のコードを実行すると、W&B ダッシュボードに新しい Table が作成され、データが表示されます。右上のパネルセレクターから `2D Projection` を選択すると、Embedding を 2 次元でプロットすることができます。スマートデフォルトが自動で選択されますが、ギアアイコンから設定メニューを開いて簡単にカスタマイズすることも可能です。この例では、使用可能な 5 つの数値次元すべてを自動的に使用しています。

{{< img src="/images/app_ui/weave_hello_world.png" alt="2D projection example" >}}

### Digits MNIST

先ほどの例は Embedding の基本的なログ方法でしたが、通常はさらに多くの次元とサンプルを扱います。ここでは MNIST Digits データセット（[UCI ML 手書き数字データセット](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits)[s](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits)）を [SciKit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) 経由で使用します。このデータセットには 1797 件の記録があり、各サンプルには 64 次元のデータが含まれています。10 クラス分類問題です。入力データを画像として可視化することもできます。

```python
import wandb
from sklearn.datasets import load_digits

with wandb.init(project="embedding_tutorial") as run:

  # データセットの読み込み
  ds = load_digits(as_frame=True)
  df = ds.data

  # "target" カラムの追加
  df["target"] = ds.target.astype(str)
  cols = df.columns.tolist()
  df = df[cols[-1:] + cols[:-1]]

  # "image" カラムの追加
  df["image"] = df.apply(
      lambda row: wandb.Image(row[1:].values.reshape(8, 8) / 16.0), axis=1
  )
  cols = df.columns.tolist()
  df = df[cols[-1:] + cols[:-1]]

  run.log({"digits": df})
```

上記のコードを実行すると、再び UI 上に Table が表示されます。`2D Projection` を選ぶと、Embedding の定義、色付け、アルゴリズム（PCA, UMAP, t-SNE）、アルゴリズムパラメータ、さらにオーバーレイ（この例では点をホバーすると画像を表示）などを設定できます。ここでも「スマートデフォルト」が有効なので、`2D Projection` を 1 クリックするだけでほぼ同じ結果が得られます。([この Embedding チュートリアル例とインタラクションする](https://wandb.ai/timssweeney/embedding_tutorial/runs/k6guxhum?workspace=user-timssweeney))。

{{< img src="/images/weave/embedding_projector.png" alt="MNIST digits projection" >}}

## ログの形式

Embeddings は様々な形式でログすることができます：

1. **埋め込み単一カラム:** データがすでに「行列」形式の場合によく使われます。この場合、1 カラムに `list[int]`, `list[float]`, `np.ndarray` のいずれかで値を格納できます。
2. **数値カラムを複数:** 上記２つの例のように、各次元ごとにカラムを作る方法です。セルの値として python の `int` または `float` をサポートしています。

{{< img src="/images/weave/logging_options.png" alt="Single embedding column" >}}
{{< img src="/images/weave/logging_option_image_right.png" alt="Multiple numeric columns" >}}

さらに、すべての Table 同様、Table の作成方法も多数あります：

1. **データフレームから直接** `wandb.Table(dataframe=df)` を使用
2. **データリストから直接** `wandb.Table(data=[...], columns=[...])`
3. **1 行ずつ追加しながらインクリメンタルに構築**（コード中でループを使う場合に便利）。`table.add_data(...)` で行を追加します
4. **埋め込みカラムを追加**（Embeddings 形式の予測リストがある場合に便利）：`table.add_col("col_name", ...)`
5. **計算カラムの追加**（関数やモデルを Table 全体にマップしたい場合に便利）：`table.add_computed_columns(lambda row, ndx: {"embedding": model.predict(row)})`

## プロットのオプション

`2D Projection` を選択後、ギアアイコンをクリックすることで描画設定を編集できます。対象となるカラム選択（前述）に加え、注目したいアルゴリズムとそのパラメータも指定できます。下図は UMAP および t-SNE それぞれのパラメータ設定画面例です。

{{< img src="/images/weave/plotting_options_left.png" alt="UMAP parameters" >}}
{{< img src="/images/weave/plotting_options_right.png" alt="t-SNE parameters" >}}

{{% alert %}}
注意：現在、全てのアルゴリズムでランダム抽出した最大 1000 行・50 次元までにダウンサンプリングしています。
{{% /alert %}}