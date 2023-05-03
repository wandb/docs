import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Scikit-Learn

wandb を使用して、わずか数行のコードで scikit-learn モデルの性能を視覚化し、比較できます。[**例を試す →**](http://wandb.me/scikit-colab)

## :fire: はじめに

### Wandb にサインアップしてログインする

a) 無料アカウントに[**サインアップ**](https://wandb.ai/site)する

b) `wandb` ライブラリを pip インストールする

c) トレーニングスクリプトでログインするには、www.wandb.ai でアカウントにサインインしておく必要があります。次に、**[**認証ページ**](https://wandb.ai/authorize)で API キーが見つかります。**

Weights and Biases を初めて使用する場合は、[クイックスタート](../../quickstart.md) をチェックしてみてください。

<Tabs
  defaultValue="cli"
  values={[
    {label: 'コマンドライン', value: 'cli'},
    {label: 'ノートブック', value: 'notebook'},
  ]}>
  <TabItem value="cli">

```
pip install wandb

wandb login
```
</TabItem>
  <TabItem value="notebook">

```python
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

# 時間経過とともにメトリクスをログする場合は、wandb.logを使ってください
wandb.log({"accuracy": accuracy})

# または、トレーニングの最後に最終的なメトリックをログするには、wandb.summaryを使ってもいいです
wandb.summary["accuracy"] = accuracy
```
### プロットを作成する

#### ステップ1：wandbをインポートして、新しいrunを開始する。

```python
import wandb
wandb.init(project="visualize-sklearn")
```

#### ステップ2：個別のプロットを可視化する

モデルのトレーニングと予測を行った後、wandbでプロットを生成して予測を分析できます。サポートされているプロットの完全なリストについては、以下の **Supported Plots** セクションを参照してください。

```python
# 単一のプロットを可視化する
wandb.sklearn.plot_confusion_matrix(y_true, y_pred, labels)
```

#### または、一度にすべてのプロットを可視化する

W&Bには、`plot_classifier` のような関数があり、いくつかの関連するプロットを表示します。

```python
# すべての分類器プロットを可視化する
wandb.sklearn.plot_classifier(clf, X_train, X_test, y_train, y_test, y_pred, y_probas, labels,
                                                         model_name='SVC', feature_names=None)

# すべての回帰プロット
wandb.sklearn.plot_regressor(reg, X_train, X_test, y_train, y_test,  model_name='Ridge')
# すべてのクラスタリングプロット
wandb.sklearn.plot_clusterer(kmeans, X_train, cluster_labels, labels=None, model_name='KMeans')
```

**または、既存のmatplotlibプロットを表示:**

Matplotlibで作成したプロットも、W＆Bダッシュボードにログして表示することができます。それを行うには、まず`plotly`をインストールする必要があります。

```
pip install plotly
```

最後に、以下のようにして、プロットをW&Bのダッシュボードにログすることができます。

```python
import matplotlib.pyplot as plt
import wandb
wandb.init(project="visualize-sklearn")

# ここで plt.plot(), plt.scatter() などをすべて行います。
# ...

# plt.show() の代わりに以下を行います:
wandb.log({"plot": plt})
```

### サポートされているプロット

#### 学習曲線
![](/images/integrations/scikit_learning_curve.png)

異なる長さのデータセットでモデルをトレーニングし、トレーニングセットとテストセットの両方に対して、データセットのサイズに対するクロスバリデーションスコアをプロットします。

`wandb.sklearn.plot_learning_curve(model, X, y)`

* model (clf or reg): フィッティング済みの回帰モデルまたは分類モデルを指定します。
* X (arr): データセットの特徴。
* y (arr): データセットのラベル。

#### ROC

![](/images/integrations/scikit_roc.png)

ROC曲線は、真陽性率（y軸）と偽陽性率（x軸）をプロットします。理想的なスコアはTPR = 1およびFPR = 0で、左上の点です。通常、ROC曲線の下の領域（AUC-ROC）を計算し、AUC-ROCが大きいほど良いとされます。

`wandb.sklearn.plot_roc(y_true, y_probas, labels)`

* y\_true (arr): テストセットのラベル。
* y\_probas (arr): テストセットの予測確率。
* labels (list): ターゲット変数（y）の名前付きラベル。

#### クラスの比率

![](/images/integrations/scikic_class_props.png)

トレーニングセットとテストセットのターゲットクラスの分布をプロットします。クラスの不均衡を検出し、あるクラスがモデルに過剰な影響を与えないことを確認するのに役立ちます。

`wandb.sklearn.plot_class_proportions(y_train, y_test, ['dog', 'cat', 'owl'])`
* y_train (arr): トレーニングセットのラベル。
* y_test (arr): テストセットのラベル。
* labels (list): 目標変数（y）の名前付きラベル。

#### PR曲線

![](/images/integrations/scikit_precision_recall.png)

異なる閾値に対する適合率と再現率のトレードオフを計算します。曲線下の面積が大きいほど、高い再現率と高い適合率が得られます。高い適合率は低い偽陽性率と関連しており、高い再現率は低い偽陰性率と関連しています。

両方のスコアが高い場合、分類器は正確な結果（高い精度）を返すだけでなく、すべての陽性結果の大部分も返します（高い再現率）。クラスが非常に不均衡な場合、PR曲線は役立ちます。

`wandb.sklearn.plot_precision_recall(y_true, y_probas, labels)`

* y\_true (arr): テストセットのラベル。
* y\_probas (arr): テストセットの予測確率。
* labels (list): 目標変数（y）の名前付きラベル。

#### 特徴量の重要度

![](/images/integrations/scikit_feature_importances.png)

分類タスクにおいて、各特徴量の重要度を評価しプロットします。`feature_importances_` 属性を持つ分類器（例：木）でのみ機能します。

`wandb.sklearn.plot_feature_importances(model, ['width', 'height, 'length'])`

* model (clf): フィット済みの分類器を入力します。
* feature\_names (list): 特徴量の名前。特徴量のインデックスを対応する名前に置き換えて、プロットを読みやすくします。
#### キャリブレーションカーブ

![](/images/integrations/scikit_calibration_curve.png)

分類器の予測確率がどれだけキャリブレーションされているかをプロットし、キャリブレーションされていない分類器をキャリブレーションする方法を比較します。ベースラインのロジスティック回帰モデルによる推定された予測確率、引数として渡されたモデル、および両方の等温キャリブレーションとシグモイドキャリブレーションによる予測確率を比較します。

キャリブレーションカーブが対角線に近いほど良いです。転置されたシグモイドのようなカーブは、過学習した分類器を表していますが、シグモイドなカーブは、学習不足の分類器を表しています。モデルの等温キャリブレーションとシグモイドキャリブレーションをトレーニングし、カーブを比較することで、モデルが過学習または学習不足であるかどうかを判断し、どのキャリブレーション（シグモイドまたは等温）がそれを修正するのに役立つかを判断できます。

詳細については、[sklearnのドキュメント](https://scikit-learn.org/stable/auto\_examples/calibration/plot\_calibration\_curve.html)を参照してください。

`wandb.sklearn.plot_calibration_curve(clf, X, y, 'RandomForestClassifier')`

* モデル（clf）： トレーニング済みの分類器を入力します。
* X（arr）： トレーニングセットの特徴。
* y（arr）： トレーニングセットのラベル。
* model\_name (str)：モデル名。デフォルトは 'Classifier' です。

#### 混同行列

![](/images/integrations/scikit_confusion_matrix.png)

混同行列を計算して分類の精度を評価します。モデルの予測の品質を評価し、モデルが間違える予測のパターンを見つけるのに役立ちます。対角線はモデルが正しく予測したものを表し、すなわち実際のラベルが予測したラベルに等しい場合です。

`wandb.sklearn.plot_confusion_matrix(y_true, y_pred, labels)`

* y\_true（arr）：テストセットのラベル。
* y\_pred（arr）：テストセットの予測ラベル。
* labels（list）：ターゲット変数（y）の名前付けラベル。

#### サマリーメトリクス
![](/images/integrations/scikit_summary_metrics.png)

回帰と分類アルゴリズムの両方に対して、要約メトリクス（分類の場合はf1、精度、適合率、再現率、回帰の場合はmse、mae、r2スコアなど）を計算します。

`wandb.sklearn.plot_summary_metrics(model, X_train, y_train, X_test, y_test)`

* model（clfまたはreg）：フィッティングされた回帰器または分類器を受け取ります。
* X（arr）：トレーニングセットの特徴量。
* y（arr）：トレーニングセットのラベル。
  * X\_test（arr）：テストセットの特徴量。
* y\_test（arr）：テストセットのラベル。

#### エルボープロット

![](/images/integrations/scikit_elbow_plot.png)

クラスターの数に関する分散の説明率とトレーニング時間を測定し、プロットします。最適なクラスター数を選択する際に役立ちます。

`wandb.sklearn.plot_elbow_curve(model, X_train)`

* model（clusterer）：フィット済みのクラスタリングモデルを受け取ります。
* X（arr）：トレーニングセットの特徴量。

#### シルエットプロット

![](/images/integrations/scikit_silhouette_plot.png)

1つのクラスタ内の各点が隣接するクラスタ内の点にどれだけ近いかを測定し、プロットします。クラスタの厚みは、クラスタのサイズに対応しています。垂直線は、すべての点の平均シルエットスコアを表します。

シルエット係数が+1に近い場合、サンプルは隣接するクラスタから遠く離れていることを示します。0の値は、サンプルが2つの隣接するクラスタ間の決定境界上またはそれに非常に近いことを示し、負の値は、それらのサンプルが誤って割り当てられたクラスタにある可能性があることを示します。
一般的には、すべてのシルエットクラスタースコアが平均値以上（赤線を超える）で、できるだけ1に近い値が望ましいです。また、クラスターのサイズもデータの基本的なパターンを反映していることが好ましいです。

`wandb.sklearn.plot_silhouette(model, X_train, ['spam', 'not spam'])`

* model (clusterer): 構築済みのクラスタリングモデルを入力します。
* X (arr): トレーニングセットの特徴量。
  * cluster\_labels (list): クラスターラベルの名前。クラスターインデックスを対応する名前に置き換えることで、プロットが読みやすくなります。

#### 外れ値候補プロット

![](/images/integrations/scikit_outlier_plot.png)

クックの距離を用いてデータポイントが回帰モデルに与える影響を測定します。影響が大きく偏ったデータは、外れ値の可能性があります。外れ値検出に役立ちます。

`wandb.sklearn.plot_outlier_candidates(model, X, y)`

* model (regressor): 構築済みの分類器を入力します。
* X (arr): トレーニングセットの特徴量。
* y (arr): トレーニングセットのラベル。

#### 残差プロット

![](/images/integrations/scikit_residuals_plot.png)

予測された目標値（y軸）と実際の目標値との差（x軸）を測定・プロットし、残差誤差の分布も表示します。

一般的に、適合度の高いモデルの残差はランダムに分布するはずです。なぜなら、良いモデルはデータセット内のほとんどの現象を説明できるため、ランダムな誤差を除いて残差は小さくなります。

`wandb.sklearn.plot_residuals(model, X, y)`
* モデル（回帰器）：フィット済みの分類器を受け取ります。

* X（arr）：トレーニングセットの特徴量。

* y（arr）：トレーニングセットのラベル。



    ご質問があれば、[Slackコミュニティ](http://wandb.me/slack)でお答えします。



## 例



* [Colabで実行](http://wandb.me/scikit-colab)：はじめに簡単なノートブックを用意しました