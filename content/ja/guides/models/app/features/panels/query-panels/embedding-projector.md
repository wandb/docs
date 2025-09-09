---
title: オブジェクトを埋め込む
description: W&B の Embedding Projector は、ユーザーが PCA、UMAP、t-SNE などの一般的な次元削減アルゴリズムを使って、多次元の埋め込みを
  2D 平面上にプロットできるようにします。
menu:
  default:
    identifier: ja-guides-models-app-features-panels-query-panels-embedding-projector
    parent: query-panels
---

{{< img src="/images/weave/embedding_projector.png" alt="埋め込みプロジェクター" >}}

[埋め込み](https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture) は、オブジェクト（人、画像、投稿、単語など）を数値のリストで表現するために使われます — ときには _ベクトル_ と呼ばれます。機械学習やデータサイエンスのユースケースでは、幅広いアプリケーションで多様な手法により埋め込みを生成できます。本ページは、埋め込みの基本を理解しており、W&B の中でそれらを可視的に分析したい読者を想定しています。

## 埋め込みの例

- [ライブ インタラクティブ デモ Report](https://wandb.ai/timssweeney/toy_datasets/reports/Feature-Report-W-B-Embeddings-Projector--VmlldzoxMjg2MjY4?accessToken=bo36zrgl0gref1th5nj59nrft9rc4r71s53zr2qvqlz68jwn8d8yyjdz73cqfyhq) 
- [サンプル Colab](https://colab.research.google.com/drive/1DaKL4lZVh3ETyYEM1oJ46ffjpGs8glXA#scrollTo=D--9i6-gXBm_).

### Hello World

W&B では、`wandb.Table` クラスを使って埋め込みをログできます。以下は、5 次元で構成された埋め込みを 3 つ記録する例です:

```python
import wandb

with wandb.init(project="embedding_tutorial") as run:
  embeddings = [
      # D1   D2   D3   D4   D5
      [0.2, 0.4, 0.1, 0.7, 0.5],  # 埋め込み 1
      [0.3, 0.1, 0.9, 0.2, 0.7],  # 埋め込み 2
      [0.4, 0.5, 0.2, 0.2, 0.1],  # 埋め込み 3
  ]
  run.log(
      {"embeddings": wandb.Table(columns=["D1", "D2", "D3", "D4", "D5"], data=embeddings)}
  )
  run.finish()
```

上のコードを実行すると、W&B ダッシュボードにあなたのデータを含む新しいテーブルが作成されます。右上のパネルセレクタから `2D Projection` を選ぶと、埋め込みを 2 次元にプロットできます。スマートデフォルトが自動で選択されますが、歯車アイコンをクリックして開く設定メニューで簡単に上書きできます。この例では、利用可能な数値次元 5 つすべてを自動で使用しています。

{{< img src="/images/app_ui/weave_hello_world.png" alt="2D プロジェクションの例" >}}

### Digits MNIST

上の例は埋め込みをログする基本的な流れを示しましたが、実際にはもっと多くの次元やサンプルを扱うことが一般的です。ここでは [SciKit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) で利用できる MNIST Digits データセット（[UCI ML hand-written digits dataset](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits)[s](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits)）を見てみましょう。このデータセットには 1,797 件のレコードがあり、それぞれ 64 次元を持ちます。課題は 10 クラスの分類ユースケースです。可視化のために入力データを画像に変換することもできます。

```python
import wandb
from sklearn.datasets import load_digits

with wandb.init(project="embedding_tutorial") as run:

  # データセットを読み込む
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

上記のコードを実行すると、UI に再びテーブルが表示されます。`2D Projection` を選ぶと、埋め込みの定義、色付け、アルゴリズム（PCA、UMAP、t-SNE）、そのパラメータ、さらにはオーバーレイ（この例では、点にホバーしたときに画像を表示）まで設定できます。今回のケースでは、これらはすべて「スマートデフォルト」で、`2D Projection` を 1 回クリックするだけで非常に近い結果が得られるはずです。（[この埋め込みチュートリアル例を操作してみる](https://wandb.ai/timssweeney/embedding_tutorial/runs/k6guxhum?workspace=user-timssweeney)）。

{{< img src="/images/weave/embedding_projector.png" alt="MNIST digits のプロジェクション" >}}

## ログのオプション

埋め込みは複数の形式でログできます:

1. **単一の埋め込みカラム:** データがすでに「行列」風の形式になっていることがよくあります。この場合は、セルの値のデータ型を `list[int]`、`list[float]`、または `np.ndarray` とする単一の埋め込みカラムを作成できます。
2. **数値カラムを複数:** 上の 2 つの例ではこの方法を使い、各次元ごとにカラムを作成しています。セルには Python の `int` または `float` を受け付けます。

{{< img src="/images/weave/logging_options.png" alt="単一の埋め込みカラム" >}}
{{< img src="/images/weave/logging_option_image_right.png" alt="複数の数値カラム" >}}

さらに、他のテーブル同様、テーブルの組み立て方にも多くの選択肢があります:

1. **dataframe** から直接: `wandb.Table(dataframe=df)`
2. **データのリスト** から直接: `wandb.Table(data=[...], columns=[...])`
3. テーブルを **行ごとに段階的に** 構築（ループがあるコードに最適）: `table.add_data(...)` で行を追加
4. テーブルに **埋め込みカラム** を追加（埋め込み形式の予測リストがある場合に最適）: `table.add_col("col_name", ...)`
5. **計算カラム** を追加（テーブル全体に適用したい関数やモデルがある場合に最適）: `table.add_computed_columns(lambda row, ndx: {"embedding": model.predict(row)})`

## プロットのオプション

`2D Projection` を選んだら、歯車アイコンをクリックして描画設定を編集できます。対象のカラム選択（上記参照）に加えて、関心のあるアルゴリズム（およびそのパラメータ）も選べます。下図はそれぞれ UMAP と t-SNE のパラメータです。

{{< img src="/images/weave/plotting_options_left.png" alt="UMAP のパラメータ" >}} 
{{< img src="/images/weave/plotting_options_right.png" alt="t-SNE のパラメータ" >}}

{{% alert %}}
注意: 現在、いずれのアルゴリズムでも、行は 1000 行、次元は 50 次元のランダムなサブセットにダウンサンプリングします。
{{% /alert %}}