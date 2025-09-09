---
title: Scikit-Learn
menu:
  default:
    identifier: ja-guides-integrations-scikit
    parent: integrations
weight: 380
---

数行のコードだけで、wandb を使って scikit-learn のモデルのパフォーマンスを可視化・比較できます。 [Try an example →](https://wandb.me/scikit-colab)

## はじめに

### サインアップして API キーを作成

API キーは、あなたのマシンを W&B に対して認証します。API キーはユーザー プロフィールから発行できます。

{{% alert %}}
よりスムーズに行うには、[W&B authorization page](https://wandb.ai/authorize) に直接アクセスして API キーを発行してください。表示された API キーをコピーし、パスワード マネージャーなどの安全な場所に保存します。
{{% /alert %}}

1. 右上のユーザー プロフィール アイコンをクリックします。
1. **User Settings** を選び、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックします。表示された API キーをコピーします。API キーを非表示にするにはページを再読み込みしてください.

### `wandb` ライブラリをインストールしてログイン

ローカルに `wandb` ライブラリをインストールしてログインするには:

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. `WANDB_API_KEY` [environment variable]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) にあなたの API キーを設定します。

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

{{% tab header="Python ノートブック" value="notebook" %}}

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

wandb.init(project="visualize-sklearn") as run:

  y_pred = clf.predict(X_test)
  accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)

  # 時系列でメトリクスを記録する場合は run.log を使います
  run.log({"accuracy": accuracy})

  # あるいは、トレーニングの最後に最終メトリクスだけを記録するなら run.summary も使えます
  run.summary["accuracy"] = accuracy
```

### プロットを作成

#### ステップ 1: wandb をインポートして新しい run を初期化

```python
import wandb

run = wandb.init(project="visualize-sklearn")
```

#### ステップ 2: プロットを可視化

#### 個別プロット

モデルをトレーニングして予測を作成したら、wandb で予測を分析するためのプロットを生成できます。対応しているチャートの一覧は下の「Supported Plots」セクションをご覧ください。

```python
# 単一のプロットを可視化
wandb.sklearn.plot_confusion_matrix(y_true, y_pred, labels)
```

#### すべてのプロット

W&B には、`plot_classifier` のように複数の関連プロットをまとめて描画する関数があります:

```python
# 分類器のプロットをすべて可視化
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

# 回帰のプロットをすべて
wandb.sklearn.plot_regressor(reg, X_train, X_test, y_train, y_test, model_name="Ridge")

# クラスタリングのプロットをすべて
wandb.sklearn.plot_clusterer(
    kmeans, X_train, cluster_labels, labels=None, model_name="KMeans"
)

run.finish()
```

#### 既存の Matplotlib プロット

Matplotlib で作成したプロットも W&B のダッシュボードにログできます。そのためには、まず `plotly` をインストールする必要があります。

```bash
pip install plotly
```

最後に、次のようにプロットを W&B のダッシュボードにログできます:

```python
import matplotlib.pyplot as plt
import wandb

with wandb.init(project="visualize-sklearn") as run:

  # ここで plt.plot() や plt.scatter() などを実行します。
  # ...

  # plt.show() の代わりに次を実行します:
  run.log({"plot": plt})
```

## 対応しているプロット

### 学習曲線

{{< img src="/images/integrations/scikit_learning_curve.png" alt="Scikit-learn の学習曲線" >}}

さまざまな長さのデータセットでモデルをトレーニングし、トレーニング セットとテストセットの両方について、交差検証スコアをデータセット サイズに対してプロットします。

`wandb.sklearn.plot_learning_curve(model, X, y)`

* model (clf or reg): 学習済みの回帰器または分類器を受け取ります。
* X (arr): データセットの特徴量。
* y (arr): データセットのラベル。

### ROC

{{< img src="/images/integrations/scikit_roc.png" alt="Scikit-learn の ROC 曲線" >}}

ROC 曲線は真陽性率 (y 軸) と偽陽性率 (x 軸) をプロットします。理想的なスコアは TPR = 1、FPR = 0 で、左上の点になります。一般に ROC 曲線下面積 (AUC-ROC) を計算し、AUC-ROC が大きいほど良いとされます。

`wandb.sklearn.plot_roc(y_true, y_probas, labels)`

* y_true (arr): テストセットのラベル。
* y_probas (arr): テストセットの予測確率。
* labels (list): 目的変数 (y) のラベル名。

### クラス比率

{{< img src="/images/integrations/scikic_class_props.png" alt="Scikit-learn のクラス比率" >}}

トレーニング セットとテストセットにおける目的クラスの分布をプロットします。不均衡データの検出や、特定のクラスがモデルに過度の影響を与えていないかの確認に役立ちます。

`wandb.sklearn.plot_class_proportions(y_train, y_test, ['dog', 'cat', 'owl'])`

* y_train (arr): トレーニング セットのラベル。
* y_test (arr): テストセットのラベル。
* labels (list): 目的変数 (y) のラベル名。

### Precision recall curve

{{< img src="/images/integrations/scikit_precision_recall.png" alt="Scikit-learn の Precision-Recall 曲線" >}}

さまざまな閾値に対する precision と recall のトレードオフを計算します。曲線下面積が大きいほど、high recall かつ high precision であることを示します。high precision は偽陽性率が低いこと、high recall は偽陰性率が低いことに対応します。

両方が高いということは、分類器が正確な結果 (高い precision) を返しつつ、陽性の大半 (高い recall) も取りこぼしていないことを意味します。PR 曲線は、クラスが非常に不均衡なときに有用です。

`wandb.sklearn.plot_precision_recall(y_true, y_probas, labels)`

* y_true (arr): テストセットのラベル。
* y_probas (arr): テストセットの予測確率。
* labels (list): 目的変数 (y) のラベル名。

### 特徴量の重要度

{{< img src="/images/integrations/scikit_feature_importances.png" alt="Scikit-learn の特徴量重要度チャート" >}}

分類タスクにおける各特徴量の重要度を評価してプロットします。木系モデルのように `feature_importances_` 属性を持つ分類器でのみ動作します。

`wandb.sklearn.plot_feature_importances(model, ['width', 'height, 'length'])`

* model (clf): 学習済みの分類器を受け取ります。
* feature_names (list): 特徴量名。インデックスを名前に置き換えることでプロットの可読性が向上します。

### キャリブレーション曲線

{{< img src="/images/integrations/scikit_calibration_curve.png" alt="Scikit-learn のキャリブレーション曲線" >}}

分類器の予測確率がどれだけ校正されているか、また未校正の分類器をどのように校正するかをプロットします。ベースラインのロジスティック回帰モデル、引数として渡したモデル、およびそのアイソトニック キャリブレーションとシグモイド キャリブレーションによる推定予測確率を比較します。

キャリブレーション曲線が対角線に近いほど良好です。転置したシグモイドのような曲線は過学習した分類器を、シグモイドのような曲線は学習不足の分類器を表します。モデルのアイソトニック版とシグモイド版を学習して曲線を比較することで、モデルが過学習か学習不足か、そしてどちらのキャリブレーション (シグモイドまたはアイソトニック) が有効かを判断できます。

詳細は [sklearn's docs](https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html) を参照してください。

`wandb.sklearn.plot_calibration_curve(clf, X, y, 'RandomForestClassifier')`

* model (clf): 学習済みの分類器を受け取ります。
* X (arr): トレーニング セットの特徴量。
* y (arr): トレーニング セットのラベル。
* model_name (str): モデル名。既定は 'Classifier'。

### 混同行列

{{< img src="/images/integrations/scikit_confusion_matrix.png" alt="Scikit-learn の混同行列" >}}

分類の精度を評価するために混同行列を計算します。モデルの予測の質を評価し、誤分類のパターンを見つけるのに役立ちます。対角成分はモデルが正しく予測したもの (実ラベルと予測ラベルが一致) を表します。

`wandb.sklearn.plot_confusion_matrix(y_true, y_pred, labels)`

* y_true (arr): テストセットのラベル。
* y_pred (arr): テストセットの予測ラベル。
* labels (list): 目的変数 (y) のラベル名。

### 集約メトリクス

{{< img src="/images/integrations/scikit_summary_metrics.png" alt="Scikit-learn のサマリーメトリクス" >}}

- 分類に対するサマリーメトリクスを計算します。例: `mse`、`mae`、`r2` スコア。
- 回帰に対するサマリーメトリクスを計算します。例: `f1`、accuracy、precision、recall。

`wandb.sklearn.plot_summary_metrics(model, X_train, y_train, X_test, y_test)`

* model (clf or reg): 学習済みの回帰器または分類器を受け取ります。
* X (arr): トレーニング セットの特徴量。
* y (arr): トレーニング セットのラベル。
  * X_test (arr): テストセットの特徴量。
* y_test (arr): テストセットのラベル。

### エルボープロット

{{< img src="/images/integrations/scikit_elbow_plot.png" alt="Scikit-learn のエルボープロット" >}}

クラスター数に対する説明分散率とトレーニング時間を測定・プロットします。最適なクラスター数の選択に役立ちます。

`wandb.sklearn.plot_elbow_curve(model, X_train)`

* model (clusterer): 学習済みのクラスタラーを受け取ります。
* X (arr): トレーニング セットの特徴量。

### シルエットプロット

{{< img src="/images/integrations/scikit_silhouette_plot.png" alt="Scikit-learn のシルエットプロット" >}}

あるクラスター内の各点が隣接クラスターの点とどれだけ近いかを測定・プロットします。クラスターの厚みはクラスター サイズに対応します。縦線は全サンプルの平均シルエットスコアを表します。

シルエット係数が +1 に近いほど、そのサンプルは隣接クラスターから十分に離れています。0 は 2 つの隣接クラスターの決定境界上またはごく近くにあることを示し、負の値は誤ったクラスターに割り当てられた可能性を示します。

一般に、すべてのクラスターのシルエットスコアが平均より高く (赤い線より右) なり、できるだけ 1 に近いことが望ましいです。また、データの基礎的なパターンを反映したクラスター サイズが好まれます。

`wandb.sklearn.plot_silhouette(model, X_train, ['spam', 'not spam'])`

* model (clusterer): 学習済みのクラスタラーを受け取ります。
* X (arr): トレーニング セットの特徴量。
  * cluster_labels (list): クラスター ラベル名。インデックスを名前に置き換えることでプロットの可読性が向上します。

### 外れ値候補プロット

{{< img src="/images/integrations/scikit_outlier_plot.png" alt="Scikit-learn の外れ値プロット" >}}

Cook の距離を使って、各データ点が回帰モデルに与える影響を測定します。影響度が極端に偏っているサンプルは外れ値の可能性があります。外れ値検出に有用です。

`wandb.sklearn.plot_outlier_candidates(model, X, y)`

* model (regressor): 学習済みの分類器を受け取ります。
* X (arr): トレーニング セットの特徴量。
* y (arr): トレーニング セットのラベル。

### 残差プロット

{{< img src="/images/integrations/scikit_residuals_plot.png" alt="Scikit-learn の残差プロット" >}}

予測されたターゲット値 (y 軸) と、実測値と予測値の差 (x 軸) をプロットし、残差誤差の分布も示します。

一般に、当てはまりの良いモデルの残差はランダムに分布します。良いモデルは、ランダム誤差を除くほとんどの現象をデータセット内で説明できるためです。

`wandb.sklearn.plot_residuals(model, X, y)`

* model (regressor): 学習済みの分類器を受け取ります。
* X (arr): トレーニング セットの特徴量。
*   y (arr): トレーニング セットのラベル。

    ご不明な点があれば、ぜひ私たちの [slack community](https://wandb.me/slack) でご質問ください。

## 例

* [Run in colab](https://wandb.me/scikit-colab): すぐに始められるシンプルなノートブック。