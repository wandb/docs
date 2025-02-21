---
title: Scikit-Learn
menu:
  default:
    identifier: ja-guides-integrations-scikit
    parent: integrations
weight: 380
---

wandb を使うと、わずか数行のコードで scikit-learn モデルのパフォーマンスを視覚化して比較できます。[**例を試す →**](http://wandb.me/scikit-colab)

## はじめに

### サインアップして API キーを作成する

API キーは、あなたのマシンを W&B に認証します。ユーザープロフィールから API キーを生成できます。

{{% alert %}}
より合理的なアプローチとして、[https://wandb.ai/authorize](https://wandb.ai/authorize) に直接アクセスして API キーを生成できます。表示された API キーをコピーし、パスワードマネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上のユーザープロフィールアイコンをクリックします。
1. **User Settings** を選択し、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックします。表示された API キーをコピーします。API キーを非表示にするには、ページを再読み込みしてください。

### `wandb` ライブラリをインストールしてログインする

ローカルに `wandb` ライブラリをインストールし、ログインするには:

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) をあなたの API キーに設定します。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb` ライブラリをインストールしてログインします。

    ```shell
    pip install wandb

    wandb login
    ```

{{% /tab %}}

{{% tab header="Python" value="python" %}}

```bash
pip install wandb
```
```python
import wandb
wandb.login()
```

{{% /tab %}}

{{% tab header="Python notebook" value="notebook" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

### メトリクスをログする

```python
import wandb

wandb.init(project="visualize-sklearn")

y_pred = clf.predict(X_test)
accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)

# メトリクスを時間の経過とともにログする場合は wandb.log を使用します
wandb.log({"accuracy": accuracy})

# OR トレーニングの最後に最終的なメトリクスをログする場合、wandb.summary を使用することもできます
wandb.summary["accuracy"] = accuracy
```

### プロットを作成する

#### ステップ 1: wandb をインポートし、新しい run を初期化する

```python
import wandb

wandb.init(project="visualize-sklearn")
```

#### ステップ 2: プロットを視覚化する

#### 個々のプロット

モデルのトレーニングと予測を行った後、wandb でプロットを生成して予測を分析することができます。サポートされているチャートの完全なリストについては、下記の **Supported Plots** セクションを参照してください。

```python
# 単一プロットを視覚化する
wandb.sklearn.plot_confusion_matrix(y_true, y_pred, labels)
```

#### すべてのプロット

W&B には `plot_classifier` などの関数があり、いくつかの関連するプロットを作成します。

```python
# すべての分類器プロットを視覚化する
wandb.sklearn.plot_classifier(
    clf,
    X_train,
    X_test,
    y_train,
    y_test,
    y_pred,
    y_probas,
    labels,
    model_name="SVC",
    feature_names=None,
)

# すべての回帰プロット
wandb.sklearn.plot_regressor(reg, X_train, X_test, y_train, y_test, model_name="Ridge")

# すべてのクラスターリングプロット
wandb.sklearn.plot_clusterer(
    kmeans, X_train, cluster_labels, labels=None, model_name="KMeans"
)
```

#### 既存の Matplotlib プロット

Matplotlib で作成されたプロットも W&B ダッシュボードにログすることができます。そのためには、まず `plotly` をインストールする必要があります。

```bash
pip install plotly
```

最後に、プロットを W&B のダッシュボードに次のようにしてログできます。

```python
import matplotlib.pyplot as plt
import wandb

wandb.init(project="visualize-sklearn")

# ここですべての plt.plot(), plt.scatter() などを行います。
# ...

# plt.show() の代わりに次の行を実行します:
wandb.log({"plot": plt})
```

## サポートされているプロット

### 学習曲線

{{< img src="/images/integrations/scikit_learning_curve.png" alt="" >}}

異なる長さのデータセットでモデルをトレーニングし、データセットサイズに対する交差検証されたスコアのプロットを生成します。トレーニングセットとテストセットの両方についてです。

`wandb.sklearn.plot_learning_curve(model, X, y)`

* model (clf または reg): フィットされた回帰モデルまたは分類器を受け取ります。
* X (arr): データセットの特徴。
* y (arr): データセットのラベル。

### ROC

{{< img src="/images/integrations/scikit_roc.png" alt="" >}}

ROC 曲線は、真陽性率 (y 軸) と偽陽性率 (x 軸) をプロットします。理想的なスコアは TPR = 1 および FPR = 0 で、左上の点です。通常、ROC 曲線の下の面積 (AUC-ROC) を計算し、AUC-ROC が大きいほど良いです。

`wandb.sklearn.plot_roc(y_true, y_probas, labels)`

* y_true (arr): テストセットのラベル。
* y_probas (arr): テストセットの予測確率。
* labels (list): 目標変数 (y) のターゲットラベル名。

### クラス比率

{{< img src="/images/integrations/scikic_class_props.png" alt="" >}}

トレーニングセットとテストセットのターゲットクラスの分布をプロットします。不均衡クラスを検出し、1 つのクラスがモデルに過度の影響を与えないようにするために便利です。

`wandb.sklearn.plot_class_proportions(y_train, y_test, ['dog', 'cat', 'owl'])`

* y_train (arr): トレーニングセットのラベル。
* y_test (arr): テストセットのラベル。
* labels (list): 目標変数 (y) のターゲットラベル名。

### 精度-再現率曲線

{{< img src="/images/integrations/scikit_precision_recall.png" alt="" >}}

異なる閾値に対する精度と再現率のトレードオフを計算します。曲線下の面積が大きいほど、高い再現率と高い精度を示します。高い精度は低い偽陽性率に関連し、高い再現率は低い偽陰性率に関連します。

両方のスコアが高ければ、分類器が正確な結果 (高精度) を返すだけでなく、大半の陽性結果 (高再現率) も返していることを示します。クラスが非常に不均衡な場合、PR 曲線は有用です。

`wandb.sklearn.plot_precision_recall(y_true, y_probas, labels)`

* y_true (arr): テストセットのラベル。
* y_probas (arr): テストセットの予測確率。
* labels (list): 目標変数 (y) のターゲットラベル名。

### 特徴重要度

{{< img src="/images/integrations/scikit_feature_importances.png" alt="" >}}

分類タスクの各特徴の重要性を評価し、プロットします。`feature_importances_` 属性を持つ分類器でのみ動作します。たとえば、ツリーなどです。

`wandb.sklearn.plot_feature_importances(model, ['width', 'height, 'length'])`

* model (clf): フィットされた分類器を受け取ります。
* feature_names (list): 特徴の名前。プロットを読みやすくするために、特徴インデックスを対応する名前に置き換えます。

### キャリブレーション曲線

{{< img src="/images/integrations/scikit_calibration_curve.png" alt="" >}}

分類器の予測確率がどれほどキャリブレーションされているかをプロットし、非キャリブレーションされた分類器をキャリブレーションする方法を示します。ベースラインのロジスティック回帰モデル、引数として渡されたモデル、および isotonic キャリブレーションとシグモイドキャリブレーションにより予測された確率を比較します。

キャリブレーション曲線が対角線に近いほど良いです。トランスポーズされたシグモイドのような曲線は過学習した分類器を示し、シグモイドのような曲線は学習不足の分類器を示します。モデルの isotonic およびシグモイドキャリブレーションをトレーニングし、それらの曲線を比較することで、モデルが過学習または学習不足かどうかを判断し、もしそうであればどちらのキャリブレーション（シグモイドまたは isotonic）が役立つか判断できます。

詳細については、[sklearnのドキュメント](https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html)を参照してください。

`wandb.sklearn.plot_calibration_curve(clf, X, y, 'RandomForestClassifier')`

* model (clf): フィットされた分類器を受け取ります。
* X (arr): トレーニングセットの特徴。
* y (arr): トレーニングセットのラベル。
* model_name (str): モデル名。デフォルトは 'Classifier'。

### 混同行列

{{< img src="/images/integrations/scikit_confusion_matrix.png" alt="" >}}

混同行列を計算して分類の精度を評価します。モデルの予測の質を評価し、モデルが誤って予測したパターンを見つけるのに役立ちます。対角線は、実際のラベルが予測ラベルと等しい場所など、モデルが正しく予測したものを表します。

`wandb.sklearn.plot_confusion_matrix(y_true, y_pred, labels)`

* y_true (arr): テストセットのラベル。
* y_pred (arr): テストセットの予測ラベル。
* labels (list): 目標変数 (y) のターゲットラベル名。

### サマリーメトリクス

{{< img src="/images/integrations/scikit_summary_metrics.png" alt="" >}}

- `mse`、`mae`、`r2` スコアなどの分類用のサマリーメトリクスを計算します。
- 回帰に対して `f1`、精度、精密度、再現率などのサマリーメトリクスを計算します。

`wandb.sklearn.plot_summary_metrics(model, X_train, y_train, X_test, y_test)`

* model (clf または reg): フィットされた回帰モデルまたは分類器を受け取ります。
* X (arr): トレーニングセットの特徴。
* y (arr): トレーニングセットのラベル。
  * X_test (arr): テストセットの特徴。
* y_test (arr): テストセットのラベル。

### エルボープロット

{{< img src="/images/integrations/scikit_elbow_plot.png" alt="" >}}

クラスター数の関数として説明される分散の割合を、トレーニング時間と共に測定してプロットします。最適なクラスター数を選択するのに有用です。

`wandb.sklearn.plot_elbow_curve(model, X_train)`

* model (clusterer): フィットされたクラスタラーを受け取ります。
* X (arr): トレーニングセットの特徴。

### シルエットプロット

{{< img src="/images/integrations/scikit_silhouette_plot.png" alt="" >}}

クラスタ内の各点が隣接クラスターの点にどれだけ近いかを測定し、プロットします。クラスターの厚さはクラスターのサイズに対応します。垂直線はすべての点の平均シルエットスコアを表します。

シルエット係数が +1 に近い場合は、サンプルが隣接クラスターから遠く離れていることを示します。0 の場合は、サンプルが 2 つの隣接クラスターの間の決定境界上または非常に近いことを示し、負の値は、これらのサンプルが誤ってクラスターに割り当てられた可能性があることを示します。

一般的にはすべてのシルエットクラスタースコアが平均以上（赤い線を超えて）で、可能な限り 1 に近いものが望ましいです。また、クラスターサイズがデータの基礎パターンを反映することが望ましいです。

`wandb.sklearn.plot_silhouette(model, X_train, ['spam', 'not spam'])`

* model (clusterer): フィットされたクラスタラーを受け取ります。
* X (arr): トレーニングセットの特徴。
  * cluster_labels (list): クラスタラベルの名前。プロットを読みやすくするために、クラスタインデックスを対応する名前に置き換えます。

### 外れ値候補プロット

{{< img src="/images/integrations/scikit_outlier_plot.png" alt="" >}}

回帰モデルへのデータポイントの影響を Cook の距離を使って測定します。大きく偏った影響を持つインスタンスは外れ値である可能性があります。外れ値検出に有用です。

`wandb.sklearn.plot_outlier_candidates(model, X, y)`

* model (regressor): フィットされた回帰モデルを受け取ります。
* X (arr): トレーニングセットの特徴。
* y (arr): トレーニングセットのラベル。

### 残差プロット

{{< img src="/images/integrations/scikit_residuals_plot.png" alt="" >}}

予測目標値 (y 軸) と実際の目標値と予測目標値の違い (x 軸) を測定し、プロットします。また、残差の誤差の分布も示します。

一般に、うまく適合したモデルの残差はランダムに分布しているべきです。なぜなら、良いモデルはデータセットにおけるほとんどの現象を説明しますが、ランダムな誤差以外を除くからです。

`wandb.sklearn.plot_residuals(model, X, y)`

* model (regressor): フィットされた回帰モデルを受け取ります。
* X (arr): トレーニングセットの特徴。
*   y (arr): トレーニングセットのラベル。

もし質問があれば、[slack コミュニティ](http://wandb.me/slack)でお答えしたいと思っています。

## 例

* [Colab で実行する](http://wandb.me/scikit-colab): 開始するための簡単なノートブック