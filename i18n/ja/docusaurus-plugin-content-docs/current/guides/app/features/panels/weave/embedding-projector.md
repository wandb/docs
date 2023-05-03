---
description: >-
  W&BのEmbedding Projectorは、PCA、UMAP、およびt-SNEなどの一般的な次元削減アルゴリズムを使用して、2D平面上に多次元の埋め込みをプロットすることができます。
---

# 埋め込みプロジェクター

![](/images/weave/embedding_projector.png)

[埋め込み](https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture)は、オブジェクト（人物、画像、投稿、単語など）を一連の数値（ベクトルとも呼ばれる）で表現するために使用されます。機械学習やデータサイエンスのユースケースでは、さまざまなアプローチで埋め込みを生成し、アプリケーション全体で使用することができます。このページでは、W&B内で埋め込みを視覚的に分析することに興味があるという前提で、読者は埋め込みについて熟知しているとしています。

## 埋め込みの例

[Live Interactive Demo Report](https://wandb.ai/timssweeney/toy\_datasets/reports/Feature-Report-W-B-Embeddings-Projector--VmlldzoxMjg2MjY4?accessToken=bo36zrgl0gref1th5nj59nrft9rc4r71s53zr2qvqlz68jwn8d8yyjdz73cqfyhq)で表示できるか、[Example Colab](https://colab.research.google.com/drive/1DaKL4lZVh3ETyYEM1oJ46ffjpGs8glXA#scrollTo=D--9i6-gXBm\_)でこのレポートからコードを実行してください。

### ハローワールド

W&Bでは、 `wandb.Table` クラスを使用して、埋め込みをログに記録することができます。以下に3つの埋め込みがありますが、それぞれに5次元が含まれています。

```python
import wandb
wandb.init(project="embedding_tutorial")
embeddings = [
    # D1   D2   D3   D4   D5
    [0.2, 0.4, 0.1, 0.7, 0.5], # embedding 1
    [0.3, 0.1, 0.9, 0.2, 0.7], # embedding 2
    [0.4, 0.5, 0.2, 0.2, 0.1], # embedding 3
]
wandb.log({
    "embeddings": wandb.Table(
        columns = ["D1", "D2", "D3", "D4", "D5"], 
        data    = embeddings
    )
})
wandb.finish()
```
上記のコードを実行すると、W&Bダッシュボードにデータが含まれる新しいTableが表示されます。右上のパネルセレクタから `2D Projection` を選択して、2次元で埋め込みをプロットすることができます。スマートデフォルトが自動的に選択されますが、歯車アイコンをクリックして設定メニューにアクセスすることで簡単に上書きできます。この例では、利用可能な5つの数値次元すべてを自動的に使用します。

![](/images/app_ui/weave_hello_world.png)

### Digits MNIST

上記の例では、埋め込みのログの基本的なメカニズムを示していますが、通常はもっと多くの次元とサンプルで作業しています。MNIST Digitsデータセット（[UCI ML手書き数字データセット](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits)[s](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits)）を[SciKit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load\_digits.html)を介して利用可能にしましょう。このデータセットには1797件の記録があり、それぞれに64の次元があります。問題は10クラス分類のユースケースです。また、入力データを画像に変換して可視化することもできます。

```python
import wandb
from sklearn.datasets import load_digits

wandb.init(project="embedding_tutorial")

# データセットをロード
ds = load_digits(as_frame=True)
df = ds.data

# "target"列を作成
df["target"] = ds.target.astype(str)
cols = df.columns.tolist()
df = df[cols[-1:] + cols[:-1]]

# "image"列を作成
df["image"] = df.apply(lambda row: wandb.Image(row[1:].values.reshape(8, 8) / 16.0), axis=1)
cols = df.columns.tolist()
df = df[cols[-1:] + cols[:-1]]

wandb.log({"digits": df})
wandb.finish()
```
上記のコードを実行すると、再びUIでテーブルが表示されます。`2D Projection` を選択することで、埋め込みの定義、色付け、アルゴリズム（PCA、UMAP、t-SNE）、アルゴリズムパラメータ、さらにオーバーレイ（この場合、点の上にマウスを置くと画像が表示される）を設定できます。この特定のケースでは、すべてが「スマートデフォルト」となっており、`2D Projection`を1回クリックするだけで、非常によく似たものが表示されます。([こちらをクリックして](https://wandb.ai/timssweeney/embedding\_tutorial/runs/k6guxhum?workspace=user-timssweeney)、この例と対話してください)。

![](/images/weave/embedding_projector.png)

## ロギングオプション

埋め込みをいくつかの異なるフォーマットでログに記録できます。

1. **単一の埋め込み列:** おそらく、データはすでに「行列」のような形式で存在していることが多いでしょう。この場合、セル値のデータタイプが `list[int]`、`list[float]`、または `np.ndarray` である1つの埋め込み列を作成できます。

2. **複数の数値列:** 上記の2つの例では、このアプローチを使用し、各次元ごとに列を作成しています。セルにはPythonの `int` または `float` を使用します。

![Single Embedding Column](/images/weave/logging_options.png)

![Many Numeric Columns](/images/weave/logging_option_image_right.png)

さらに、すべてのテーブルと同様に、テーブルの作成方法についても多くのオプションがあります。

1. **データフレーム**を使用して直接 `wandb.Table(dataframe=df)`

2. **データのリスト**から直接 `wandb.Table(data=[...], columns=[...])`

3. **逐次的に行ごとに**テーブルを構築する（コードにループがある場合は便利）：`table.add_data(...)` を使ってテーブルに行を追加します。

4. 埋め込み列をテーブルに追加する（埋め込みの形式で予測のリストがある場合は便利）：`table.add_col("col_name", ...)`

5. **計算された列**を追加する（テーブルにマップしたい関数やモデルがある場合は便利）：`table.add_computed_columns(lambda row, ndx: {"embedding": model.predict(row)})`

## プロットオプション

`2D Projection` を選択した後、歯車アイコンをクリックしてレンダリング設定を編集できます。上記に示すように、目的の列を選択することに加えて、興味を持っているアルゴリズム（および所望のパラメータ）を選択できます。以下に、UMAP とt-SNEのパラメータがそれぞれ示されています。

![](/images/weave/plotting_options_left.png)

![](/images/weave/plotting_options_right.png)

:::info

注: 現在、すべての3つのアルゴリズムについて、ランダムな1000行と50次元のサブセットにダウンサンプリングしています。

:::