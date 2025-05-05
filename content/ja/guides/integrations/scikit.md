---
title: Scikit-Learn
menu:
  default:
    identifier: ja-guides-integrations-scikit
    parent: integrations
weight: 380
---

wandbを使って、scikit-learn モデルの性能を数行のコードで視覚化し比較することができます。 [**例を試す →**](http://wandb.me/scikit-colab)

## 始めに

### サインアップしてAPIキーを作成

APIキーは、あなたのマシンをW&Bに認証するためのものです。ユーザーのプロフィールからAPIキーを生成できます。

{{% alert %}}
よりスムーズな方法として、[https://wandb.ai/authorize](https://wandb.ai/authorize)に直接アクセスしてAPIキーを生成することができます。表示されたAPIキーをコピーし、パスワードマネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上のユーザープロフィールアイコンをクリック。
2. **User Settings** を選択し、**API Keys** セクションまでスクロール。
3. **Reveal** をクリック。表示されたAPIキーをコピーします。APIキーを非表示にするには、ページを再読み込みしてください。

### `wandb` ライブラリをインストールしてログイン

ローカルで`wandb` ライブラリをインストールし、ログインするには:

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) をあなたのAPIキーに設定します。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb` ライブラリをインストールし、ログインします。

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

# メトリクスを時間でログする場合、wandb.logを使用
wandb.log({"accuracy": accuracy})

# またはトレーニングの最後にメトリクスをログするには、wandb.summaryを使用することもできます
wandb.summary["accuracy"] = accuracy
```

### プロットを作成する

#### ステップ1: wandbをインポートして新しいrunを初期化

```python
import wandb

wandb.init(project="visualize-sklearn")
```

#### ステップ2: プロットを可視化する

#### 個別のプロット

モデルをトレーニングし、予測を行った後、wandbでプロットを生成して予測を分析することができます。サポートされているチャートの完全なリストについては、以下の**Supported Plots**セクションを参照してください。

```python
# 単一のプロットを可視化
wandb.sklearn.plot_confusion_matrix(y_true, y_pred, labels)
```

#### すべてのプロット

W&B には `plot_classifier` などの関数があり、関連する複数のプロットを描画します。

```python
# すべての分類器プロットを可視化
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

# すべてのクラスタリングプロット
wandb.sklearn.plot_clusterer(
    kmeans, X_train, cluster_labels, labels=None, model_name="KMeans"
)
```

#### 既存のMatplotlibプロット

Matplotlibで作成されたプロットも、W&B ダッシュボードにログすることができます。そのためには、最初に `plotly` をインストールする必要があります。

```bash
pip install plotly
```

最後に、以下のようにW&Bのダッシュボードにプロットをログすることができます。

```python
import matplotlib.pyplot as plt
import wandb

wandb.init(project="visualize-sklearn")

# plt.plot(), plt.scatter() などをここで行います。
# ...

# plt.show()の代わりに:
wandb.log({"plot": plt})
```

## サポートされているプロット

### 学習曲線

{{< img src="/images/integrations/scikit_learning_curve.png" alt="" >}}

モデルを様々な長さのデータセットでトレーニングし、交差検証スコアとデータセットサイズのプロットを生成します。トレーニングセットとテストセット両方に対して。

`wandb.sklearn.plot_learning_curve(model, X, y)`

* model (clf or reg): 学習済みの回帰器または分類器を受け取ります。
* X (arr): データセットの特徴。
* y (arr): データセットのラベル。

### ROC

{{< img src="/images/integrations/scikit_roc.png" alt="" >}}

ROC曲線は、真陽性率 (y軸) 対 偽陽性率 (x軸) をプロットします。理想的なスコアは、TPR = 1 かつ FPR = 0で、グラフの左上の点です。通常、ROC曲線の下面積 (AUC-ROC) を計算し、AUC-ROC が大きいほど良いです。

`wandb.sklearn.plot_roc(y_true, y_probas, labels)`

* y_true (arr): テストセットのラベル。
* y_probas (arr): テストセットの予測確率。
* labels (list): 目標変数 (y) の名前付きラベル。

### クラスの割合

{{< img src="/images/integrations/scikic_class_props.png" alt="" >}}

トレーニングセットとテストセット内のターゲットクラスの分布をプロットします。非バランスなクラスを検出し、1つのクラスがモデルに過度の影響を与えないようにするために役立ちます。

`wandb.sklearn.plot_class_proportions(y_train, y_test, ['dog', 'cat', 'owl'])`

* y_train (arr): トレーニングセットのラベル。
* y_test (arr): テストセットのラベル。
* labels (list): 目標変数 (y) の名前付きラベル。

### 精度-再現率曲線

{{< img src="/images/integrations/scikit_precision_recall.png" alt="" >}}

異なる閾値に対する精度と再現率のトレードオフを計算します。曲線下面積が高いということは、再現率も精度も高いことを表しており、高精度は低誤報率に、高再現率は低漏れ率に関連しています。

精度と再現率の両方が高いことは、分類器が正確な結果（高精度）を返していること、さらに全ての陽性結果の大半を返していること（高再現率）を示しています。クラスが非常に不均衡な時に、PR曲線は役立ちます。

`wandb.sklearn.plot_precision_recall(y_true, y_probas, labels)`

* y_true (arr): テストセットのラベル。
* y_probas (arr): テストセットの予測確率。
* labels (list): 目標変数 (y) の名前付きラベル。

### 特徴の重要度

{{< img src="/images/integrations/scikit_feature_importances.png" alt="" >}}

分類タスクにおける各特徴の重要度を評価しプロットします。ツリーのような `feature_importances_` 属性を持つ分類器でのみ動作します。

`wandb.sklearn.plot_feature_importances(model, ['width', 'height', 'length'])`

* model (clf): 学習済みの分類器を受け取ります。
* feature_names (list): 特徴の名前。プロット中の特徴のインデックスを対応する名前で置き換えることで読みやすくします。

### キャリブレーション曲線

{{< img src="/images/integrations/scikit_calibration_curve.png" alt="" >}}

分類器の予測確率がどれだけキャリブレーションされているか、そしてどのように未キャリブレーションの分類器をキャリブレーションするかをプロットします。ロジスティック回帰ベースラインモデル、引数として渡されたモデル、およびそのアイソトニックキャリブレーションとシグモイドキャリブレーションによって、推定された予測確率を比較します。

キャリブレーション曲線が対角線に近いほど良好です。転写されたシグモイド型の曲線は過適合した分類器を表し、シグモイド型の曲線は学習不足の分類器を表します。モデルのアイソトニックおよびシグモイドキャリブレーションをトレーニングし、その曲線を比較することで、モデルがオーバーフィットかアンダーフィットしているかを判断し、どのキャリブレーション（シグモイドまたはアイソトニック）が問題を修正するのに役立つかを理解できます。

詳細については、[sklearnのドキュメント](https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html)を参照してください。

`wandb.sklearn.plot_calibration_curve(clf, X, y, 'RandomForestClassifier')`

* model (clf): 学習済みの分類器を受け取ります。
* X (arr): トレーニングセットの特徴。
* y (arr): トレーニングセットのラベル。
* model_name (str): モデル名。デフォルトは'Classifier'です。

### 混同行列

{{< img src="/images/integrations/scikit_confusion_matrix.png" alt="" >}}

分類の精度を評価するために混同行列を計算します。モデルの予測の質を評価し、モデルが間違ってしまう予測のパターンを見つけるのに役立ちます。対角線は、実際のラベルと予測ラベルが一致する正しい予測を表します。

`wandb.sklearn.plot_confusion_matrix(y_true, y_pred, labels)`

* y_true (arr): テストセットのラベル。
* y_pred (arr): テストセットの予測ラベル。
* labels (list): 目標変数 (y) の名前付きラベル。

### サマリーメトリクス

{{< img src="/images/integrations/scikit_summary_metrics.png" alt="" >}}

- `mse`、`mae`、`r2`スコアなどの分類のサマリーメトリクスを計算します。
- `f1`、精度、再現率などの回帰のサマリーメトリクスを計算します。

`wandb.sklearn.plot_summary_metrics(model, X_train, y_train, X_test, y_test)`

* model (clf or reg): 学習済みの回帰器または分類器を受け取ります。
* X (arr): トレーニングセットの特徴。
* y (arr): トレーニングセットのラベル。
  * X_test (arr): テストセットの特徴。
* y_test (arr): テストセットのラベル。

### エルボープロット

{{< img src="/images/integrations/scikit_elbow_plot.png" alt="" >}}

クラスターの数に対する分散の説明率をトレーニング時間とともに測定しプロットします。クラスター数の最適値を選ぶのに役立ちます。

`wandb.sklearn.plot_elbow_curve(model, X_train)`

* model (clusterer): 学習済みのクラスタリングアルゴリズムを受け取ります。
* X (arr): トレーニングセットの特徴。

### シルエットプロット

{{< img src="/images/integrations/scikit_silhouette_plot.png" alt="" >}}

1つのクラスター内の各ポイントが、隣接するクラスターポイントにどれだけ近いかを測定しプロットします。クラスターの厚みはクラスターサイズに対応します。垂直線は全ポイントの平均シルエットスコアを示します。

+1に近いシルエット係数は、サンプルが隣接クラスターから遠いことを示します。0の値は、サンプルが隣接クラスター間の意思決定境界にあることを示しています。負の値は、これらのサンプルが誤ってクラスターに割り当てられた可能性があることを示します。

一般的に私たちは、すべてのシルエットクラスター スコアが、平均以上（赤線を超えたところ）そして1にできるだけ近いことを望みます。また、データ中の基礎パターンを反映したクラスターサイズを好みます。

`wandb.sklearn.plot_silhouette(model, X_train, ['spam', 'not spam'])`

* model (clusterer): 学習済みのクラスタリングアルゴリズムを受け取ります。
* X (arr): トレーニングセットの特徴。
  * cluster_labels (list): クラスターラベルの名前。プロット中のクラスターインデックスを対応する名前で置き換え、読みやすくします。

### 外れ値候補プロット

{{< img src="/images/integrations/scikit_outlier_plot.png" alt="" >}}

Cookの距離を使用して、回帰モデルの各データポイントの影響を評価します。大きく偏った影響を持つインスタンスは外れ値である可能性があります。外れ値検出に役立ちます。

`wandb.sklearn.plot_outlier_candidates(model, X, y)`

* model (regressor): 学習済みの分類器を受け取ります。
* X (arr): トレーニングセットの特徴。
* y (arr): トレーニングセットのラベル。

### 残差プロット

{{< img src="/images/integrations/scikit_residuals_plot.png" alt="" >}}

予測された目標値 (y軸) 対 実際の目標値と予測された目標値の差 (x軸) 、さらに残差誤差の分布を測定しプロットします。

一般的に、適切にフィットされたモデルの残差はランダムに分布しているべきです。というのも、良いモデルは、データセット中のほとんどの現象を説明するからです。ランダムな誤差を除いて。

`wandb.sklearn.plot_residuals(model, X, y)`

* model (regressor): 学習済みの分類器を受け取ります。
* X (arr): トレーニングセットの特徴。
* y (arr): トレーニングセットのラベル。

ご質問がある場合は、私たちの[slackコミュニティ](http://wandb.me/slack)でお答えしますので、お気軽にどうぞ。

## 例

* [コラボで実行](http://wandb.me/scikit-colab): 始めるためのシンプルなノートブック
