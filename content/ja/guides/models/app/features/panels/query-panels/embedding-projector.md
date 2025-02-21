---
title: Embed objects
description: W&B の Embedding Projector を使用すると、PCA、UMAP、t-SNE などの一般的な次元削減アルゴリズムを使用して、多次元埋め込みを
  2D 平面にプロットできます。
menu:
  default:
    identifier: ja-guides-models-app-features-panels-query-panels-embedding-projector
    parent: query-panels
---

{{< img src="/images/weave/embedding_projector.png" alt="" >}}

[埋め込み](https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture) は、オブジェクト（人、画像、投稿、単語など）を数値のリスト（_ベクトル_ とも呼ばれる）で表現するために使用されます。 機械学習 とデータサイエンスの ユースケース では、埋め込みは、さまざまなアプリケーションにわたるさまざまなアプローチを使用して生成できます。このページでは、読者が埋め込みについてよく理解しており、W&B 内でそれらを視覚的に分析することに関心があると想定しています。

## 埋め込みの例

- [ライブインタラクティブ デモ Report](https://wandb.ai/timssweeney/toy_datasets/reports/Feature-Report-W-B-Embeddings-Projector--VmlldzoxMjg2MjY4?accessToken=bo36zrgl0gref1th5nj59nrft9rc4r71s53zr2qvqlz68jwn8d8yyjdz73cqfyhq)
- [Colab の例](https://colab.research.google.com/drive/1DaKL4lZVh3ETyYEM1oJ46ffjpGs8glXA#scrollTo=D--9i6-gXBm_)。

### Hello World

W&B では、`wandb.Table` クラスを使用して埋め込みを ログ に記録できます。それぞれが 5 次元で構成される 3 つの埋め込みの次の例を考えてみましょう。

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

上記の コード を実行すると、W&B ダッシュボード にデータを含む新しい テーブル が表示されます。右上の パネル セレクターから [2D Projection] を選択して、埋め込みを 2 次元でプロットできます。スマート デフォルト が自動的に選択されます。これは、歯車アイコンをクリックしてアクセスできる 設定 メニューで簡単にオーバーライドできます。この例では、利用可能な 5 つの数値次元すべてを自動的に使用します。

{{< img src="/images/app_ui/weave_hello_world.png" alt="" >}}

### Digits MNIST

上記の例は、埋め込みを ログ に記録する基本的なメカニズムを示していますが、通常は、より多くの次元とサンプルを扱っています。 MNIST Digits データセット ([UCI ML 手書き数字データセット](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits)[s](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits)) ([SciKit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)経由で利用可能) について考えてみましょう。このデータセットには 1797 個のレコードがあり、それぞれに 64 次元があります。この問題は、10 クラス分類の ユースケース です。入力データを画像に変換して、可視化することもできます。

```python
import wandb
from sklearn.datasets import load_digits

wandb.init(project="embedding_tutorial")

# データセットをロードする
ds = load_digits(as_frame=True)
df = ds.data

# 「target」列を作成する
df["target"] = ds.target.astype(str)
cols = df.columns.tolist()
df = df[cols[-1:] + cols[:-1]]

# 「image」列を作成する
df["image"] = df.apply(
    lambda row: wandb.Image(row[1:].values.reshape(8, 8) / 16.0), axis=1
)
cols = df.columns.tolist()
df = df[cols[-1:] + cols[:-1]]

wandb.log({"digits": df})
wandb.finish()
```

上記の コード を実行すると、UI に テーブル が再び表示されます。[2D Projection] を選択すると、埋め込みの定義、色付け、アルゴリズム (PCA、UMAP、t-SNE)、アルゴリズム パラメータ を 設定 したり、オーバーレイ (この場合は、ポイントにカーソルを合わせると画像が表示されます) したりできます。この特定のケースでは、これらはすべて「スマート デフォルト」であり、[2D Projection] を 1 回クリックするだけで、非常によく似たものが表示されるはずです。([ここをクリックして](https://wandb.ai/timssweeney/embedding_tutorial/runs/k6guxhum?workspace=user-timssweeney)この例を操作してください)。

{{< img src="/images/weave/embedding_projector.png" alt="" >}}

## ログ オプション

埋め込みは、さまざまな形式で ログ に記録できます。

1.  **単一の埋め込み列:** 多くの場合、データはすでに「行列」のような形式になっています。この場合、単一の埋め込み列を作成できます。ここで、セル値の データ 型は `list[int]`、`list[float]`、または `np.ndarray` になります。
2.  **複数の数値列:** 上記の 2 つの例では、このアプローチを使用し、次元ごとに列を作成します。現在、セルには python `int` または `float` を使用できます。

{{< img src="/images/weave/logging_options.png" alt="Single Embedding Column" >}}
{{< img src="/images/weave/logging_option_image_right.png" alt="Many Numeric Columns" >}}

さらに、すべての テーブル と同様に、テーブル の構築方法に関して多くのオプションがあります。

1.  `wandb.Table(dataframe=df)` を使用して **dataframe** から直接
2.  `wandb.Table(data=[...], columns=[...])` を使用して **データ のリスト** から直接
3.  テーブル を **行ごとに段階的に** 構築します (コード に ループ がある場合に最適です)。 `table.add_data(...)` を使用して、テーブル に行を追加します。
4.  テーブル に **埋め込み列** を追加します (埋め込み形式の 予測 のリストがある場合に最適です)。 `table.add_col("col_name", ...)`
5.  テーブル に **計算列** を追加します (テーブル にマッピングする関数または モデル がある場合に最適です)。 `table.add_computed_columns(lambda row, ndx: {"embedding": model.predict(row)})`

## プロット オプション

[2D Projection] を選択した後、歯車アイコンをクリックしてレンダリング 設定 を編集できます。目的の列を選択するだけでなく (上記を参照)、目的のアルゴリズム (および目的の パラメータ ) を選択できます。以下に、それぞれ UMAP と t-SNE の パラメータ を示します。

{{< img src="/images/weave/plotting_options_left.png" alt="" >}}
{{< img src="/images/weave/plotting_options_right.png" alt="" >}}

{{% alert %}}
注: 現在、3 つのアルゴリズムすべてに対して、1000 行と 50 次元のランダムな サブセット に ダウンサンプリング しています。
{{% /alert %}}
