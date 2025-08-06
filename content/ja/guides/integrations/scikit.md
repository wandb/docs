---
title: Scikit-Learn
menu:
  default:
    identifier: ja-guides-integrations-scikit
    parent: integrations
weight: 380
---

wandb を使えば、ほんの数行のコードで scikit-learn モデルのパフォーマンスを可視化し比較できます。[サンプルを試す →](https://wandb.me/scikit-colab)

## はじめに

### サインアップと API キーの作成

APIキーは、お使いのマシンを W&B に認証するためのものです。API キーはユーザープロフィールから作成できます。

{{% alert %}}
よりスムーズな方法として、[W&B の認証ページ](https://wandb.ai/authorize) に直接アクセスして API キーを生成することができます。表示された API キーをコピーし、パスワードマネージャーなど安全な場所に保管してください。
{{% /alert %}}

1. 右上のユーザープロフィールアイコンをクリックします。
1. **User Settings** を選択し、**API Keys** のセクションまでスクロールします。
1. **Reveal** をクリックし、表示された API キーをコピーします。API キーを非表示にしたい場合はページをリロードしてください。

### `wandb` ライブラリのインストールとログイン

ローカル環境で `wandb` ライブラリをインストールしてログインするには：

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}})に API キーを設定します。

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

### メトリクスのログ

```python
import wandb

wandb.init(project="visualize-sklearn") as run:

  y_pred = clf.predict(X_test)
  accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)

  # 時系列でメトリクスを記録する場合は run.log を使用
  run.log({"accuracy": accuracy})

  # OR 最終的なメトリクスのみ記録したい場合は run.summary も利用可能
  run.summary["accuracy"] = accuracy
```

### プロットの作成

#### ステップ1: wandb をインポートし新しい run を初期化

```python
import wandb

run = wandb.init(project="visualize-sklearn")
```

#### ステップ2: プロットを可視化

#### 個別プロット

モデルをトレーニングして予測を行った後、wandb でプロットを生成し予測結果を分析できます。対応している全チャートのリストは下記「**Supported Plots**」セクションをご参照ください。

```python
# 単一プロットを可視化
wandb.sklearn.plot_confusion_matrix(y_true, y_pred, labels)
```

#### すべてのプロット

W&B には `plot_classifier` などの関数があり、関連する複数のプロットをまとめて可視化できます。

```python
# すべての分類プロット
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

# 全回帰プロット
wandb.sklearn.plot_regressor(reg, X_train, X_test, y_train, y_test, model_name="Ridge")

# クラスタリングプロット
wandb.sklearn.plot_clusterer(
    kmeans, X_train, cluster_labels, labels=None, model_name="KMeans"
)

run.finish()
```

#### 既存の Matplotlib プロット

Matplotlib で作成したプロットも、W&B ダッシュボードにログとして記録できます。まずはじめに `plotly` をインストールしてください。

```bash
pip install plotly
```

その後、以下のように W&B のダッシュボードへプロットを記録できます。

```python
import matplotlib.pyplot as plt
import wandb

with wandb.init(project="visualize-sklearn") as run:

  # ここで plt.plot(), plt.scatter() などを記述
  # ...

  # plt.show() の代わりに以下で記録
  run.log({"plot": plt})
```

## 対応プロット

### 学習曲線

{{< img src="/images/integrations/scikit_learning_curve.png" alt="Scikit-learn learning curve" >}}

さまざまなサイズのデータセットでモデルをトレーニングし、交差検証スコアとデータセットサイズ（学習・テストセット）の関係をプロットします。

`wandb.sklearn.plot_learning_curve(model, X, y)`

* model (clf または reg): 学習済みの回帰器または分類器を指定。
* X (配列): データセットの特徴量。
* y (配列): データセットのラベル。

### ROC 曲線

{{< img src="/images/integrations/scikit_roc.png" alt="Scikit-learn ROC curve" >}}

ROC 曲線は真陽性率（y軸）と偽陽性率（x軸）をプロットします。理想的なのは左上（TPR=1, FPR=0）の点です。通常は ROC 曲線下の面積（AUC-ROC）を算出し、AUC-ROC が大きいほど性能が高いことを示します。

`wandb.sklearn.plot_roc(y_true, y_probas, labels)`

* y_true (配列): テストセットのラベル。
* y_probas (配列): テストセットの予測確率。
* labels (リスト): ターゲット変数（y）のラベル名。

### クラス分布

{{< img src="/images/integrations/scikic_class_props.png" alt="Scikit-learn classification properties" >}}

学習セットおよびテストセットのターゲットクラスの分布をプロットします。クラス不均衡の検出や、一部クラスがモデルに過剰な影響を与えていないか確認するのに便利です。

`wandb.sklearn.plot_class_proportions(y_train, y_test, ['dog', 'cat', 'owl'])`

* y_train (配列): 学習セットのラベル。
* y_test (配列): テストセットのラベル。
* labels (リスト): ターゲット変数（y）のラベル名。

### PR 曲線 (Precision Recall curve)

{{< img src="/images/integrations/scikit_precision_recall.png" alt="Scikit-learn precision-recall curve" >}}

各しきい値ごとに precision と recall のトレードオフを算出。カーブ下の面積が大きいほど、精度と再現率の両方が高いことを意味します。精度（precision）が高いと偽陽性が少なく、再現率（recall）が高いと偽陰性が少ないです。

両方のスコアが高いことで、分類器が的確な結果（高精度）を返し、かつ陽性のほとんどを検出（高再現率）できていることを示します。PR 曲線はクラス不均衡が大きいケースで有用です。

`wandb.sklearn.plot_precision_recall(y_true, y_probas, labels)`

* y_true (配列): テストセットのラベル。
* y_probas (配列): テストセットの予測確率。
* labels (リスト): ターゲット変数（y）のラベル名。

### 特徴量の重要度

{{< img src="/images/integrations/scikit_feature_importances.png" alt="Scikit-learn feature importance chart" >}}

分類タスクにおいて、各特徴量の重要度を評価しプロットします。`feature_importances_` 属性を持つツリーモデル等に利用できます。

`wandb.sklearn.plot_feature_importances(model, ['width', 'height, 'length'])`

* model (clf): 学習済みの分類器を指定。
* feature_names (リスト): 特徴量名。インデックスの代わりに名前を表示でき、可読性が向上します。

### キャリブレーションカーブ

{{< img src="/images/integrations/scikit_calibration_curve.png" alt="Scikit-learn calibration curve" >}}

分類器が予測する確率のキャリブレーション状況（どれだけ適合しているか）をプロットします。ベースラインのロジスティック回帰や、渡したモデル、等温キャリブレーション、シグモイドキャリブレーションとの比較を行います。

カーブが対角線に近いほど良いです。sigmoid の逆転したようなカーブは過学習（overfit）を、普通の sigmoid 型は学習不足（underfit）を示します。等温キャリブレーションとシグモイドキャリブレーションの比較から過学習／学習不足かどうか、どのキャリブレーションが有効かを判断できます。

詳細については [sklearn のドキュメント](https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html) をご覧ください。

`wandb.sklearn.plot_calibration_curve(clf, X, y, 'RandomForestClassifier')`

* model (clf): 学習済みの分類器を指定。
* X (配列): 学習セットの特徴量。
* y (配列): 学習セットのラベル。
* model_name (文字列): モデル名（デフォルトは 'Classifier'）

### 混同行列

{{< img src="/images/integrations/scikit_confusion_matrix.png" alt="Scikit-learn confusion matrix" >}}

分類の精度を評価するために混同行列を計算します。モデルの予測精度や、誤った予測の傾向を分析するときに便利です。対角成分は予測が正解だったもの（実際のラベルと予測ラベルが一致したもの）を表します。

`wandb.sklearn.plot_confusion_matrix(y_true, y_pred, labels)`

* y_true (配列): テストセットのラベル。
* y_pred (配列): テストセットの予測ラベル。
* labels (リスト): ターゲット変数（y）のラベル名。

### サマリメトリクス

{{< img src="/images/integrations/scikit_summary_metrics.png" alt="Scikit-learn summary metrics" >}}

- 分類の場合: `mse`、`mae`、`r2` スコアなど主要なメトリクスを計算
- 回帰の場合: `f1`、accuracy、precision、recall など主要なメトリクスを計算

`wandb.sklearn.plot_summary_metrics(model, X_train, y_train, X_test, y_test)`

* model (clf または reg): 学習済みの回帰器または分類器を指定。
* X (配列): 学習セットの特徴量。
* y (配列): 学習セットのラベル。
  * X_test (配列): テストセットの特徴量。
* y_test (配列): テストセットのラベル。

### エルボープロット

{{< img src="/images/integrations/scikit_elbow_plot.png" alt="Scikit-learn elbow plot" >}}

クラスタ数の関数として説明される分散の割合を計測し、トレーニング時間と合わせてプロットします。最適なクラスタ数を決める際に役立ちます。

`wandb.sklearn.plot_elbow_curve(model, X_train)`

* model (clusterer): 学習済みのクラスタラを指定。
* X (配列): 学習セットの特徴量。

### シルエットプロット

{{< img src="/images/integrations/scikit_silhouette_plot.png" alt="Scikit-learn silhouette plot" >}}

各クラスタ内の点が近隣クラスタとどれくらい離れているかを測定・プロットします。クラスタの太さはクラスタサイズに対応し、垂直線は全データ点の平均シルエットスコアを示します。

シルエット係数 +1 付近は該当サンプルが他クラスタから十分離れていること、0 はクラスタ境界上またはその付近、負値は誤ったクラスタ割当の可能性を示します。

理想的には、すべてのシルエットスコアが平均線より大きく、1 に近いのが望ましいです。またクラスタサイズもデータ内のパターンをよく反映していることが望まれます。

`wandb.sklearn.plot_silhouette(model, X_train, ['spam', 'not spam'])`

* model (clusterer): 学習済みのクラスタラを指定。
* X (配列): 学習セットの特徴量。
  * cluster_labels (リスト): クラスタラベル名。クラスタインデックスの代わりに名前表示で可読性向上。

### 外れ値候補プロット

{{< img src="/images/integrations/scikit_outlier_plot.png" alt="Scikit-learn outlier plot" >}}

回帰モデルにおける Cook’s 距離で各データ点の影響度を測定・プロットします。影響度が大きく偏ったインスタンスは外れ値の可能性があり、外れ値検出に利用できます。

`wandb.sklearn.plot_outlier_candidates(model, X, y)`

* model (regressor): 学習済みの分類器を指定。
* X (配列): 学習セットの特徴量。
* y (配列): 学習セットのラベル。

### 残差プロット

{{< img src="/images/integrations/scikit_residuals_plot.png" alt="Scikit-learn residuals plot" >}}

予測ターゲット値（y軸）と、実際値と予測値の差分（x軸）、および残差エラーの分布をプロットします。

一般的に、適切にフィットしたモデルでは残差がランダムに分布します。これは、良いモデルがデータセット内のほとんどの現象を説明しており、ランダム誤差のみが残るためです。

`wandb.sklearn.plot_residuals(model, X, y)`

* model (regressor): 学習済みの分類器を指定。
* X (配列): 学習セットの特徴量。
*   y (配列): 学習セットのラベル。

何か質問があれば [slack コミュニティ](https://wandb.me/slack) でご相談ください。

## サンプル

* [colab で実行](https://wandb.me/scikit-colab): はじめてみたい方のためのシンプルなノートブックです。