---
title: オブジェクトの埋め込み
description: W&B の Embedding Projector は、PCA、UMAP、t-SNE などの一般的な次元削減アルゴリズムを使って、多次元の埋め込みを
  2 次元平面上に可視化できます。
menu:
  default:
    identifier: ja-guides-models-app-features-panels-query-panels-embedding-projector
    parent: query-panels
---

{{< img src="/images/weave/embedding_projector.png" alt="Embedding projector" >}}

[Embedding](https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture) はオブジェクト（人物、画像、投稿、単語など）を _ベクトル_ とも呼ばれる数字のリストとして表現する手法です。機械学習やデータサイエンスのユースケースでは、さまざまな方法・用途で embedding を生成できます。このページでは、embedding についての基本知識があり、W&B 内で embedding を可視的に分析したい方を対象としています。

## Embedding の例

- [ライブインタラクティブデモ Reports](https://wandb.ai/timssweeney/toy_datasets/reports/Feature-Report-W-B-Embeddings-Projector--VmlldzoxMjg2MjY4?accessToken=bo36zrgl0gref1th5nj59nrft9rc4r71s53zr2qvqlz68jwn8d8yyjdz73cqfyhq)
- [サンプル Colab](https://colab.research.google.com/drive/1DaKL4lZVh3ETyYEM1oJ46ffjpGs8glXA#scrollTo=D--9i6-gXBm_)

### Hello World

W&B では、`wandb.Table` クラスを使って embedding のログが簡単にできます。下記は、5 次元で構成された 3 つの embedding を log する例です。

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

このコードを実行すると、W&B ダッシュボードにデータを含む新しい Table が作成されます。右上のパネルセレクタで `2D Projection` を選ぶと、embedding を 2 次元にプロットできます。スマートな初期値が自動的に選択されますが、設定メニュー（歯車アイコン）から簡単に変更可能です。この例では、利用可能な 5 つの数値次元がすべて自動的に使われます。

{{< img src="/images/app_ui/weave_hello_world.png" alt="2D projection example" >}}

### Digits MNIST

上記の例で embedding の基本的なログ方法を説明しましたが、実際にはより多くの次元・サンプルを扱うケースが多いでしょう。ここでは MNIST Digits データセット（[UCI ML 手書き数字データセット](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits)[s](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits)）を使います。[SciKit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) から取得可能です。このデータセットには 1,797 件、各 64 次元のデータが含まれます。問題設定は 10 クラスの分類ユースケースです。入力データを画像にも変換し、可視化も行います。

```python
import wandb
from sklearn.datasets import load_digits

with wandb.init(project="embedding_tutorial") as run:

  # データセットの読み込み
  ds = load_digits(as_frame=True)
  df = ds.data

  # "target" カラムを作成
  df["target"] = ds.target.astype(str)
  cols = df.columns.tolist()
  df = df[cols[-1:] + cols[:-1]]

  # "image" カラムを作成
  df["image"] = df.apply(
      lambda row: wandb.Image(row[1:].values.reshape(8, 8) / 16.0), axis=1
  )
  cols = df.columns.tolist()
  df = df[cols[-1:] + cols[:-1]]

  run.log({"digits": df})
```

このコードを実行した後、同様に UI 上で Table が表示されます。`2D Projection` を選択すると、embedding の定義、色分け、アルゴリズム（PCA、UMAP、t-SNE）、そのパラメータやオーバーレイ（この例ではポイントにマウスをのせると画像表示）などが設定できます。この例ではすべて「スマートな初期値」で表示されるため、`2D Projection` を１クリックするだけで確認できます。([この embedding チュートリアル例を触ってみる](https://wandb.ai/timssweeney/embedding_tutorial/runs/k6guxhum?workspace=user-timssweeney))

{{< img src="/images/weave/embedding_projector.png" alt="MNIST digits projection" >}}

## ロギングオプション

embedding は様々なフォーマットで log できます：

1. **単一 embedding カラム:** 多くの場合、データはすでに「行列」的な形になっています。この場合、1つの embedding カラムを作成できます（セルの値は `list[int]`、`list[float]`、`np.ndarray` のいずれか）。
2. **複数の数値カラム:** 上記2例で利用した方法で、各次元ごとにカラムを作成します。現在 Python の `int` または `float` 型をサポートしています。

{{< img src="/images/weave/logging_options.png" alt="Single embedding column" >}}
{{< img src="/images/weave/logging_option_image_right.png" alt="Multiple numeric columns" >}}

さらに、他の Table と同様、Table を作成する方法もいくつか選べます：

1. **dataframe から直接**: `wandb.Table(dataframe=df)` でそのまま作成
2. **リストデータから直接**: `wandb.Table(data=[...], columns=[...])`
3. **1行ずつ追加してインクリメンタルに構築**: （ループ処理などに便利）`table.add_data(...)` で行を追加
4. **embedding カラムをテーブルに追加**: （embedding 予測リストがある場合に便利）`table.add_col("col_name", ...)`
5. **計算カラム（computed column）を追加**: （関数やモデルをテーブル全体に適用したいとき）`table.add_computed_columns(lambda row, ndx: {"embedding": model.predict(row)})`

## プロットオプション

`2D Projection` を選択した後、歯車アイコンからプロットの設定を編集できます。対象カラムの選択（上記参照）に加え、好みのアルゴリズムやそのパラメータも選択できます。下図は UMAP と t-SNE それぞれのパラメータ例です。

{{< img src="/images/weave/plotting_options_left.png" alt="UMAP parameters" >}} 
{{< img src="/images/weave/plotting_options_right.png" alt="t-SNE parameters" >}}

{{% alert %}}
注意：現在、すべてのアルゴリズムでランダムに 1,000 行・50 次元にダウンサンプルしています。
{{% /alert %}}