---
title: Embed objects
description: W&B の Embedding Projector は、 PCA、UMAP、t-SNE などの一般的な次元削減アルゴリズムを使用して、ユーザー
  が多次元の埋め込みを 2D 平面にプロットできるようにします。
menu:
  default:
    identifier: ja-guides-models-app-features-panels-query-panels-embedding-projector
    parent: query-panels
---

{{< img src="/images/weave/embedding_projector.png" alt="" >}}

[埋め込み](https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture)は、オブジェクト（人、画像、投稿、単語など）を数字のリストで表現するために使用されます。これは時折、_ベクトル_ と呼ばれます。機械学習やデータサイエンスのユースケースでは、埋め込みはさまざまなアプリケーションにわたって様々なアプローチを使用して生成できます。このページでは、読者が埋め込みに精通しており、W&Bの中でそれらを視覚的に分析することに興味を持っていることを前提としています。

## 埋め込みの例

- [ライブインタラクティブデモレポート](https://wandb.ai/timssweeney/toy_datasets/reports/Feature-Report-W-B-Embeddings-Projector--VmlldzoxMjg2MjY4?accessToken=bo36zrgl0gref1th5nj59nrft9rc4r71s53zr2qvqlz68jwn8d8yyjdz73cqfyhq) 
- [Colabの例](https://colab.research.google.com/drive/1DaKL4lZVh3ETyYEM1oJ46ffjpGs8glXA#scrollTo=D--9i6-gXBm_).

### ハローワールド

W&Bでは、`wandb.Table`クラスを使用して埋め込みをログすることができます。これは、次のように5次元からなる3つの埋め込みの例です。

```python
import wandb

wandb.init(project="embedding_tutorial")
embeddings = [
    # D1   D2   D3   D4   D5
    [0.2, 0.4, 0.1, 0.7, 0.5],  # 埋め込み1
    [0.3, 0.1, 0.9, 0.2, 0.7],  # 埋め込み2
    [0.4, 0.5, 0.2, 0.2, 0.1],  # 埋め込み3
]
wandb.log(
    {"embeddings": wandb.Table(columns=["D1", "D2", "D3", "D4", "D5"], data=embeddings)}
)
wandb.finish()
```

上記のコードを実行すると、W&Bダッシュボードにデータを含む新しいテーブルが作成されます。右上のパネルセレクターから `2D Projection` を選択すると、埋め込みを2次元でプロットできます。スマートデフォルトが自動的に選択され、設定メニューからギアアイコンをクリックして簡単に上書き可能です。この例では、利用可能な5つの数値次元すべてを自動的に使用します。

{{< img src="/images/app_ui/weave_hello_world.png" alt="" >}}

### 数字MNIST

上記の例では、埋め込みをログする基本的なメカニクスを示していますが、通常ははるかに多くの次元やサンプルを扱うことになります。ここでは、[SciKit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)を通じて利用可能なMNIST 数字データセット ([UCI ML 手書き数字データセット](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits)[s](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits))を例に考えてみましょう。このデータセットには1797件のレコードがあり、それぞれ64次元を持っています。これは10クラス分類のユースケースです。入力データを、可視化のために画像に変換することもできます。

```python
import wandb
from sklearn.datasets import load_digits

wandb.init(project="embedding_tutorial")

# データセットのロード
ds = load_digits(as_frame=True)
df = ds.data

# "target" 列の作成
df["target"] = ds.target.astype(str)
cols = df.columns.tolist()
df = df[cols[-1:] + cols[:-1]]

# "image" 列の作成
df["image"] = df.apply(
    lambda row: wandb.Image(row[1:].values.reshape(8, 8) / 16.0), axis=1
)
cols = df.columns.tolist()
df = df[cols[-1:] + cols[:-1]]

wandb.log({"digits": df})
wandb.finish()
```

上記のコードを実行すると、再びUIにテーブルが表示されます。`2D Projection` を選択することで、埋め込みの定義、カラーリング、アルゴリズム（PCA、UMAP、t-SNE）、アルゴリズムのパラメータ、さらにはオーバーレイ（この場合、ポイントにホバーした際に画像を表示）を構成できます。この特定のケースでは、これらはすべて "スマートデフォルト" で、`2D Projection` を1回クリックするだけで非常に似たものが表示されるはずです。（[こちらをクリックしてこの例をインタラクティブに操作](https://wandb.ai/timssweeney/embedding_tutorial/runs/k6guxhum?workspace=user-timssweeney)してください）。

{{< img src="/images/weave/embedding_projector.png" alt="" >}}

## ログオプション

埋め込みをさまざまな形式でログすることができます：

1. **単一の埋め込み列:** データがすでに "行列" のような形式になっていることがよくあります。この場合、データのセルの値のデータ型が `list[int]`, `list[float]`, または `np.ndarray` である単一の埋め込み列を作成することができます。
2. **多数の数値列:** 上記の2つの例では、このアプローチを使用して各次元の列を作成します。現在、セルにはPythonの`int`または`float`をサポートしています。

{{< img src="/images/weave/logging_options.png" alt="Single Embedding Column" >}}
{{< img src="/images/weave/logging_option_image_right.png" alt="Many Numeric Columns" >}}

さらに、すべてのテーブルと同様に、テーブルを構築する方法についても多くのオプションがあります：

1. **データフレーム**から直接：`wandb.Table(dataframe=df)` を使用
2. **データのリスト**から直接：`wandb.Table(data=[...], columns=[...])` を使用
3. テーブルを **インクリメンタルに行ごとに作成**（コード内にループがある場合に最適）。`table.add_data(...)` を使用してテーブルに行を追加
4. テーブルに **埋め込み列** を追加（埋め込み形式の予測リストがある場合に最適）：`table.add_col("col_name", ...)`
5. **計算された列** を追加（関数またはモデルをテーブル上にマップしたい場合に最適）：`table.add_computed_columns(lambda row, ndx: {"embedding": model.predict(row)})`

## プロットオプション

`2D Projection` を選択した後、ギアアイコンをクリックしてレンダリング設定を編集することができます。上記で示した列の選択に加えて、興味のあるアルゴリズム（および必要なパラメータ）を選択することができます。以下には、UMAPとt-SNEのパラメータをそれぞれ示しています。

{{< img src="/images/weave/plotting_options_left.png" alt="" >}} 
{{< img src="/images/weave/plotting_options_right.png" alt="" >}}

{{% alert %}}
ノート: 現在、すべての3つのアルゴリズムに対してランダムなサブセットで1000行と50次元にダウンサンプリングしています。
{{% /alert %}}