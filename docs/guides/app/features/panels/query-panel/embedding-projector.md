---
description: W&BのEmbedding Projectorを使用すると、ユーザーはPCA、UMAP、t-SNEなどの一般的な次元削減アルゴリズムを使用して、多次元の埋め込みを2D平面にプロットできます。
displayed_sidebar: default
---


# Embedding Projector

![](/images/weave/embedding_projector.png)

[Embeddings](https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture) は、オブジェクト（人、画像、投稿、単語など）を数値のリストで表現するために使用されます。これらは時々 _ベクトル_ とも呼ばれます。機械学習やデータサイエンスのユースケースでは、さまざまなアプローチで多種多様なアプリケーションからEmbeddingsを生成できます。このページでは、読者がEmbeddingsに精通しており、W&B内でそれらを視覚的に分析することに興味があることを前提としています。

## Embedding Examples

[ライブ・インタラクティブ・デモ・レポート](https://wandb.ai/timssweeney/toy\_datasets/reports/Feature-Report-W-B-Embeddings-Projector--VmlldzoxMjg2MjY4?accessToken=bo36zrgl0gref1th5nj59nrft9rc4r71s53zr2qvqlz68jwn8d8yyjdz73cqfyhq)に直接飛び込むこともできますし、このレポートのコードを[Example Colab](https://colab.research.google.com/drive/1DaKL4lZVh3ETyYEM1oJ46ffjpGs8glXA#scrollTo=D--9i6-gXBm\_)から実行することもできます。

### Hello World

W&Bでは、`wandb.Table`クラスを使用してEmbeddingsをログに記録することができます。以下の例では、5次元からなる3つのEmbeddingsを示します。

```python
import wandb

wandb.init(project="embedding_tutorial")
embeddings = [
    # D1   D2   D3   D4   D5
    [0.2, 0.4, 0.1, 0.7, 0.5],  # embedding 1
    [0.3, 0.1, 0.9, 0.2, 0.7],  # embedding 2
    [0.4, 0.5, 0.2, 0.2, 0.1],  # embedding 3
]
wandb.log(
    {"embeddings": wandb.Table(columns=["D1", "D2", "D3", "D4", "D5"], data=embeddings)}
)
wandb.finish()
```

上記のコードを実行した後、W&Bダッシュボードには新しいTableがデータとともに表示されます。右上のパネルセレクタから `2D Projection` を選択すると、Embeddingsを2次元でプロットできます。スマートデフォルトが自動的に選択され、ギアアイコンをクリックしてアクセスする設定メニューで簡単に上書きできます。この例では、利用可能な5つの数値次元すべてが自動的に使用されます。

![](/images/app_ui/weave_hello_world.png)

### Digits MNIST

上記の例ではEmbeddingの基本的なメカニクスを示しましたが、通常はもっと多くの次元とサンプルを扱います。MNIST Digitsデータセット（ [UCI ML手書き数字認識データセット](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits)）を[Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load\_digits.html)経由で利用します。このデータセットには1797件のレコードがあり、それぞれに64次元があります。この問題は10クラス分類のユースケースです。入力データを画像に変換して可視化することもできます。

```python
import wandb
from sklearn.datasets import load_digits

wandb.init(project="embedding_tutorial")

# データセットのロード
ds = load_digits(as_frame=True)
df = ds.data

# "target"カラムの作成
df["target"] = ds.target.astype(str)
cols = df.columns.tolist()
df = df[cols[-1:] + cols[:-1]]

# "image"カラムの作成
df["image"] = df.apply(
    lambda row: wandb.Image(row[1:].values.reshape(8, 8) / 16.0), axis=1
)
cols = df.columns.tolist()
df = df[cols[-1:] + cols[:-1]]

wandb.log({"digits": df})
wandb.finish()
```

上記のコードを実行した後、再びUIにTableが表示されます。`2D Projection` を選択すると、Embeddingの定義、カラーリング、アルゴリズム（PCA、UMAP、t-SNE）、アルゴリズムのパラメータ、およびオーバーレイ（この場合、点にホバーしたときに画像が表示されます）の設定ができます。この特定のケースでは、すべて「スマートデフォルト」であり、`2D Projection` をワンクリックするだけで非常に類似したものを見ることができるはずです。この例を使って[インタラクティブにクリック](https://wandb.ai/timssweeney/embedding\_tutorial/runs/k6guxhum?workspace=user-timssweeney)します。

![](/images/weave/embedding_projector.png)

## Logging Options

Embeddingsは、さまざまなフォーマットでログに記録できます。

1. **Single Embedding Column:** 多くの場合、データはすでに「マトリックス」形式になっています。この場合、単一のEmbeddingカラムを作成できます。セルの値のデータ型は `list[int]`、`list[float]` または `np.ndarray` である場合があります。
2. **Multiple Numeric Columns:** 上記の2つの例では、このアプローチを使用して、各次元のカラムを作成します。現在、セルにはPythonの `int` または `float` を受け入れます。

![Single Embedding Column](/images/weave/logging_options.png)
![Many Numeric Columns](/images/weave/logging_option_image_right.png)

さらに、すべてのテーブルと同様に、テーブルの構築方法についても多くのオプションがあります。

1. **データフレーム**から直接 `wandb.Table(dataframe=df)` を使用
2. **データのリスト**から直接 `wandb.Table(data=[...], columns=[...])` を使用
3. テーブルを**行ごとにインクリメンタルに構築**（コードにループがある場合に最適）。`table.add_data(...)` を使用してテーブルに行を追加
4. テーブルに**Embeddingカラム**を追加（予測のリストが embeddings 形式になっている場合に最適）：`table.add_col("col_name", ...)`
5. **計算カラム**を追加（関数やモデルをテーブルにマップしたい場合に最適）：`table.add_computed_columns(lambda row, ndx: {"embedding": model.predict(row)})`

## Plotting Options

`2D Projection` を選択した後、ギアアイコンをクリックしてレンダリング設定を編集できます。上記の意図されたカラムを選択するだけでなく、興味のあるアルゴリズム（および希望のパラメータ）を選択することもできます。以下にUMAPとt-SNEそれぞれのパラメータを示します。

![](/images/weave/plotting_options_left.png)
![](/images/weave/plotting_options_right.png)

:::info
Note: 現在、すべてのアルゴリズムで1000行と50次元のランダムサブセットにダウンサンプリングしています。
:::
