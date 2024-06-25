---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Scikit-Learn

wandbを利用して、scikit-learnモデルのパフォーマンスを数行のコードで可視化および比較することができます。[**例を試してみる →**](http://wandb.me/scikit-colab)

## :fire: スタートガイド

### wandbにサインアップしてログインする

a) [**Sign up**](https://wandb.ai/site) して無料アカウントを作成する

b) `wandb`ライブラリをPipでインストールする

c) トレーニングスクリプトにログインするためには、まず www.wandb.ai でアカウントにサインインしている必要があります。その後、[**Authorize page**](https://wandb.ai/authorize) で **APIキー** を確認してください。

もし初めてWeights and Biasesを利用する場合は、[quickstart](../../quickstart.md)を見ると良いでしょう。

<Tabs
  defaultValue="cli"
  values={[
    {label: 'Command Line', value: 'cli'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="cli">

```
pip install wandb

wandb login
```

  </TabItem>
  <TabItem value="notebook">

```notebook
!pip install wandb

wandb.login()
```

  </TabItem>
</Tabs>

### メトリクスのログ

```python
import wandb

wandb.init(project="visualize-sklearn")

y_pred = clf.predict(X_test)
accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)

# メトリクスを時間と共にログする場合は、 wandb.log を使用
wandb.log({"accuracy": accuracy})

# または、トレーニング終了時に最終メトリクスをログする場合は、 wandb.summary を使用
wandb.summary["accuracy"] = accuracy
```

### プロットの作成

#### ステップ1: wandbをインポートし、新しいrunを初期化する。

```python
import wandb

wandb.init(project="visualize-sklearn")
```

#### ステップ2: 個々のプロットを可視化する

モデルをトレーニングし、予測を行った後に、wandbで予測を分析するためのプロットを生成できます。サポートされているチャートの完全なリストは、以下の**サポートされるプロット**セクションを参照してください。

```python
# 単一のプロットを可視化
wandb.sklearn.plot_confusion_matrix(y_true, y_pred, labels)
```

#### またはすべてのプロットを一度に可視化する

W&Bには、`plot_classifier`のような複数の関連プロットを生成する関数があります。

```python
# すべての分類器のプロットを可視化
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

**または既存のmatplotlibプロットをプロットする:**

Matplotlibで作成されたプロットもW&Bダッシュボードにログすることができます。これを行うには、まず`plotly`をインストールする必要があります。

```
pip install plotly
```

最後に、プロットは次のようにW&Bのダッシュボードにログすることができます。

```python
import matplotlib.pyplot as plt
import wandb

wandb.init(project="visualize-sklearn")

# ここで plt.plot()、plt.scatter() などを実行します。
# ...

# plt.show()の代わりに以下を実行します:
wandb.log({"plot": plt})
```

### サポートされるプロット

#### ラーニングカーブ

![](/images/integrations/scikit_learning_curve.png)

データセットの長さを変えてモデルをトレーニングし、トレーニングセットとテストセットの両方について、交差検証スコアとデータセットサイズのプロットを生成します。

`wandb.sklearn.plot_learning_curve(model, X, y)`

* model (clf または reg): 訓練済みの回帰器または分類器を入力します。
* X (arr): データセットの特徴量。
* y (arr): データセットのラベル。

#### ROC

![](/images/integrations/scikit_roc.png)

ROC曲線は真陽性率 (y軸) と偽陽性率 (x軸) をプロットします。理想的なスコアは TPR = 1 で FPR = 0 で、これは左上の点です。通常、ROC曲線の下の領域 (AUC-ROC) を計算し、AUC-ROCが大きいほど良いとされます。

`wandb.sklearn.plot_roc(y_true, y_probas, labels)`

* y_true (arr): テストセットのラベル。
* y_probas (arr): テストセットの予測確率。
* labels (list): ターゲット変数 (y) の名前付きラベル。

#### クラスの割合

![](/images/integrations/scikic_class_props.png)

トレーニングセットとテストセットのターゲットクラスの分布をプロットします。不均衡なクラスを検出し、一つのクラスがモデルに不当な影響を与えないことを確認するために役立ちます。

`wandb.sklearn.plot_class_proportions(y_train, y_test, ['dog', 'cat', 'owl'])`

* y_train (arr): トレーニングセットのラベル。
* y_test (arr): テストセットのラベル。
* labels (list): ターゲット変数 (y) の名前付きラベル。

#### 精度 - 再現性曲線

![](/images/integrations/scikit_precision_recall.png)

異なるしきい値に対する精度と再現率のトレードオフを計算します。曲線下の領域が大きいほど、高い再現率と高い精度を兼ね備えていることを表し、高い精度は低い偽陽性率、高い再現率は低い偽陰性率に関連します。

両方が高スコアであれば、分類器が正確な結果（高精度）を返すと同時に、ほとんどの正の結果（高再現率）も返していることを示します。PR曲線はクラスが非常に不均衡な場合に有用です。

`wandb.sklearn.plot_precision_recall(y_true, y_probas, labels)`

* y_true (arr): テストセットのラベル。
* y_probas (arr): テストセットの予測確率。
* labels (list): ターゲット変数 (y) の名前付きラベル。

#### 特徴量の重要度

![](/images/integrations/scikit_feature_importances.png)

分類タスクの各特徴量の重要性を評価し、プロットします。ツリーのように `feature_importances_` 属性を持つ分類器でのみ動作します。

`wandb.sklearn.plot_feature_importances(model, ['width', 'height, 'length'])`

* model (clf): 訓練済みの分類器を入力します。
* feature_names (list): 特徴量の名前。プロットの見やすさのために、特徴量のインデックスを対応する名前で置き換えます。

#### 校正曲線

![](/images/integrations/scikit_calibration_curve.png)

分類器の予測確率がどれほど校正されているか、および未校正の分類器をどのように校正するかをプロットします。ベースラインのロジスティック回帰モデル、引数として渡されたモデル、およびそのアイソトニック校正とシグモイド校正の予測確率を比較します。

校正曲線が対角線に近いほど良いです。転置されたシグモイドのような曲線は過適応した分類器を表し、シグモイドのような曲線は学習不足の分類器を表します。モデルのアイソトニックおよびシグモイド校正をトレーニングし、それぞれの曲線を比較することで、モデルが過剰適合か学習不足かを判断し、どちらの校正（シグモイドまたはアイソトニック）が問題解決に役立つかを確認できます。

詳細は [sklearnのドキュメント](https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html) を参照してください。

`wandb.sklearn.plot_calibration_curve(clf, X, y, 'RandomForestClassifier')`

* model (clf): 訓練済みの分類器を入力します。
* X (arr): トレーニングセットの特徴。
* y (arr): トレーニングセットのラベル。
* model_name (str): モデル名。デフォルトは 'Classifier'

#### 混同行列

![](/images/integrations/scikit_confusion_matrix.png)

分類の精度を評価するために混同行列を計算します。モデル予測の質を評価し、モデルが誤る予測のパターンを見つけるのに役立ちます。対角線上はモデルが正解した予測を表し、つまり実際のラベルが予測ラベルと一致する部分です。

`wandb.sklearn.plot_confusion_matrix(y_true, y_pred, labels)`

* y_true (arr): テストセットのラベル。
* y_pred (arr): テストセットの予測されたラベル。
* labels (list): ターゲット変数 (y) の名前付きラベル。

#### サマリーメトリクス

![](/images/integrations/scikit_summary_metrics.png)

回帰および分類アルゴリズムのそれぞれのサマリーメトリクス（例えば分類の場合は f1、精度、精度および再現率、回帰の場合は mse、mae、r2 スコア）を計算します。

`wandb.sklearn.plot_summary_metrics(model, X_train, y_train, X_test, y_test)`

* model (clf または reg): 訓練済みの回帰器または分類器を入力します。
* X (arr): トレーニングセットの特徴。
* y (arr): トレーニングセットのラベル。
  * X_test (arr): テストセットの特徴。
* y_test (arr): テストセットのラベル。

#### エルボープロット

![](/images/integrations/scikit_elbow_plot.png)

クラスター数の関数として説明される分散の割合とトレーニング時間を測定してプロットします。最適なクラスター数を選ぶ際に役立ちます。

`wandb.sklearn.plot_elbow_curve(model, X_train)`

* model (clusterer): 訓練済みのクラスタリングアルゴリズムを入力します。
* X (arr): トレーニングセットの特徴。

#### シルエットプロット

![](/images/integrations/scikit_silhouette_plot.png)

クラスター内の各点が隣接するクラスターの点とどれだけ近いかを測定してプロットします。クラスターの厚さはクラスターのサイズに対応し、縦線はすべての点の平均シルエットスコアを表します。

シルエット係数が+1に近いほど、そのサンプルは隣接するクラスターから遠く離れています。値が0の場合、そのサンプルは二つの隣接するクラスターの境界上または非常に近くにあることを示し、負の値はそのサンプルが誤ってクラスターに割り当てられた可能性があることを示します。

一般に、すべてのシルエットクラスターのスコアは平均以上（赤い線を越えて）であり、可能な限り1に近いことが望ましいです。また、クラスターサイズはデータの基礎となるパターンを反映するものが好ましいです。

`wandb.sklearn.plot_silhouette(model, X_train, ['spam', 'not spam'])`

* model (clusterer): 訓練済みのクラスタリングアルゴリズムを入力します。
* X (arr): トレーニングセットの特徴。
  * cluster_labels (list): クラスターラベルの名前。プロットの見やすさのために、クラスターインデックスを対応する名前で置き換えます。

#### アウトライヤー候補プロット

![](/images/integrations/scikit_outlier_plot.png)

回帰モデルのクックの距離を使用してデータポイントの影響を測定します。影響が大きく偏ったインスタンスは異常値である可能性があります。異常値検出に役立ちます。

`wandb.sklearn.plot_outlier_candidates(model, X, y)`

* model (regressor): 訓練済みの回帰器を入力します。
* X (arr): トレーニングセットの特徴。
* y (arr): トレーニングセットのラベル。

#### 残差プロット

![](/images/integrations/scikit_residuals_plot.png)

予測されたターゲット値 (y軸) と実測値と予測値の差 (x軸) をプロットし、残差エラーの分布もプロットします。

一般に、よく適合したモデルの残差はランダムに分布するはずです。なぜなら、良いモデルはデータセットのほとんどの現象を説明し、ランダムなエラーを除外するからです。

`wandb.sklearn.plot_residuals(model, X, y)`

* model (regressor): 訓練済みの回帰器を入力します。
* X (arr): トレーニングセットの特徴。
* y (arr): トレーニングセットのラベル。

    質問がある場合は、[slack community](http://wandb.me/slack) でお答えします。

## 例

* [Run in colab](http://wandb.me/scikit-colab): 始めるのに便利なシンプルなノートブック