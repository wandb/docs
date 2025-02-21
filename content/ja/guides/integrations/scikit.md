---
title: Scikit-Learn
menu:
  default:
    identifier: ja-guides-integrations-scikit
    parent: integrations
weight: 380
---

数行のコードで、scikit-learn モデルのパフォーマンスを視覚化し、比較するために wandb を使用できます。[**サンプルを試す →**](http://wandb.me/scikit-colab)

## 始め方

### サインアップして APIキー を作成する

APIキー は、お使いのマシンを W&B に対して認証します。APIキー は、ユーザー プロファイルから生成できます。

{{% alert %}}
より効率的なアプローチとして、[https://wandb.ai/authorize](https://wandb.ai/authorize) に直接アクセスして APIキー を生成できます。表示された APIキー をコピーし、パスワード マネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上隅にあるユーザー プロファイル アイコンをクリックします。
2. **User Settings**を選択し、**API Keys**セクションまでスクロールします。
3. **Reveal**をクリックします。表示された APIキー をコピーします。APIキー を非表示にするには、ページをリロードします。

### `wandb` ライブラリをインストールしてログインする

`wandb` ライブラリをローカルにインストールしてログインするには:

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}})を APIキー に設定します。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

2. `wandb` ライブラリをインストールしてログインします。

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

### メトリクス を記録する

```python
import wandb

wandb.init(project="visualize-sklearn")

y_pred = clf.predict(X_test)
accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)

# 時間経過に伴うメトリクスを記録する場合は、wandb.log を使用します
wandb.log({"accuracy": accuracy})

# または、トレーニングの最後に最終的なメトリクスを記録するには、wandb.summary を使用することもできます
wandb.summary["accuracy"] = accuracy
```

### プロット を作成する

#### ステップ 1: wandb をインポートし、新しい run を初期化します

```python
import wandb

wandb.init(project="visualize-sklearn")
```

#### ステップ 2: プロット を視覚化する

#### 個々のプロット

モデル をトレーニングし、予測を行った後、wandb でプロットを生成して予測を分析できます。サポートされているグラフの完全なリストについては、以下の**サポートされているプロット**セクションを参照してください。

```python
# 単一のプロットを視覚化する
wandb.sklearn.plot_confusion_matrix(y_true, y_pred, labels)
```

#### すべてのプロット

W&B には、いくつかの関連するプロット をプロットする `plot_classifier` などの関数があります。

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

# すべてのクラスタリング プロット
wandb.sklearn.plot_clusterer(
    kmeans, X_train, cluster_labels, labels=None, model_name="KMeans"
)
```

#### 既存の Matplotlib プロット

Matplotlib で作成されたプロット は、W&B ダッシュボード にも記録できます。そのためには、最初に `plotly` をインストールする必要があります。

```bash
pip install plotly
```

最後に、プロット は次のように W&B のダッシュボード に記録できます。

```python
import matplotlib.pyplot as plt
import wandb

wandb.init(project="visualize-sklearn")

# ここですべての plt.plot(), plt.scatter() などを実行します。
# ...

# plt.show() を実行する代わりに、次のようにします。
wandb.log({"plot": plt})
```

## サポートされているプロット

### 学習曲線

{{< img src="/images/integrations/scikit_learning_curve.png" alt="" >}}

さまざまな長さのデータセット でモデル をトレーニングし、トレーニング セット と テストセット の両方について、データセット サイズに対するクロス検証済みスコア のプロット を生成します。

`wandb.sklearn.plot_learning_curve(model, X, y)`

* model (clf または reg): 適合された回帰子 または 分類器 を受け取ります。
* X (arr): データセット の特徴。
* y (arr): データセット のラベル。

### ROC

{{< img src="/images/integrations/scikit_roc.png" alt="" >}}

ROC 曲線は、真陽性率 (y 軸) 対 偽陽性率 (x 軸) をプロット します。理想的なスコア は TPR = 1 および FPR = 0 であり、これは左上の点です。通常、ROC 曲線下面積 (AUC-ROC) を計算し、AUC-ROC が大きいほど優れています。

`wandb.sklearn.plot_roc(y_true, y_probas, labels)`

* y_true (arr): テストセット のラベル。
* y_probas (arr): テストセット の予測確率。
* labels (list): ターゲット変数 (y) の名前付きラベル。

### クラス 割合

{{< img src="/images/integrations/scikic_class_props.png" alt="" >}}

トレーニング セット と テストセット における ターゲット クラス の分布をプロット します。不均衡なクラス を検出し、1 つのクラス がモデル に不均衡な影響を与えないようにするのに役立ちます。

`wandb.sklearn.plot_class_proportions(y_train, y_test, ['dog', 'cat', 'owl'])`

* y_train (arr): トレーニング セット のラベル。
* y_test (arr): テストセット のラベル。
* labels (list): ターゲット変数 (y) の名前付きラベル。

### 適合率 - 再現率 曲線

{{< img src="/images/integrations/scikit_precision_recall.png" alt="" >}}

さまざまなしきい値に対する適合率 と 再現率 の間のトレードオフ を計算します。曲線下面積が大きいほど、高い再現率 と 高い適合率 の両方が表されます。高い適合率 は低い偽陽性率 に関連し、高い再現率 は低い偽陰性率 に関連します。

両方の高いスコア は、分類器 が正確な結果 (高い適合率) を返し、すべての陽性結果の大部分 (高い再現率) を返していることを示しています。PR 曲線は、クラス が非常に不均衡な場合に役立ちます。

`wandb.sklearn.plot_precision_recall(y_true, y_probas, labels)`

* y_true (arr): テストセット のラベル。
* y_probas (arr): テストセット の予測確率。
* labels (list): ターゲット変数 (y) の名前付きラベル。

### 特徴 の重要度

{{< img src="/images/integrations/scikit_feature_importances.png" alt="" >}}

分類タスク における各 特徴 の重要度を評価してプロット します。ツリー のように、`feature_importances_` 属性を持つ分類器 でのみ機能します。

`wandb.sklearn.plot_feature_importances(model, ['width', 'height, 'length'])`

* model (clf): 適合された分類器 を受け取ります。
* feature_names (list): 特徴 の名前。特徴 インデックス を対応する名前に置き換えることで、プロット を読みやすくします。

### キャリブレーション 曲線

{{< img src="/images/integrations/scikit_calibration_curve.png" alt="" >}}

分類器 の予測確率 がどの程度調整されているか、および調整されていない分類器 を調整する方法をプロット します。ベースライン ロジスティック回帰モデル 、引数として渡されたモデル 、およびそのアイソトニック キャリブレーション と シグモイド キャリブレーション の両方によって推定された予測確率 を比較します。

キャリブレーション 曲線が対角線に近いほど優れています。転置されたシグモイド のような曲線は 学習過多 の分類器 を表し、シグモイド のような曲線は 学習不足 の分類器 を表します。モデル のアイソトニック キャリブレーション と シグモイド キャリブレーション をトレーニング し、それらの曲線を比較することで、モデル が 学習過多 または 学習不足 であるかどうか、およびその場合、どのキャリブレーション (シグモイド または アイソトニック) が修正に役立つかを把握できます。

詳細については、[sklearn のドキュメント](https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html) を確認してください。

`wandb.sklearn.plot_calibration_curve(clf, X, y, 'RandomForestClassifier')`

* model (clf): 適合された分類器 を受け取ります。
* X (arr): トレーニング セット の特徴。
* y (arr): トレーニング セット のラベル。
* model_name (str): モデル 名。デフォルト は 'Classifier' です

### 混同行列

{{< img src="/images/integrations/scikit_confusion_matrix.png" alt="" >}}

混同行列 を計算して、分類 の精度を評価します。モデル の予測の品質を評価し、モデル が間違っている予測のパターン を見つけるのに役立ちます。対角線は、実際のラベル が予測ラベル と等しい場合など、モデル が正しく取得した予測を表します。

`wandb.sklearn.plot_confusion_matrix(y_true, y_pred, labels)`

* y_true (arr): テストセット のラベル。
* y_pred (arr): テストセット の予測ラベル。
* labels (list): ターゲット変数 (y) の名前付きラベル。

### サマリーメトリクス

{{< img src="/images/integrations/scikit_summary_metrics.png" alt="" >}}

- `mse`、`mae`、`r2` スコア など、分類 の サマリーメトリクス を計算します。
- `f1`、精度、適合率、再現率 など、回帰 の サマリーメトリクス を計算します。

`wandb.sklearn.plot_summary_metrics(model, X_train, y_train, X_test, y_test)`

* model (clf または reg): 適合された回帰子 または 分類器 を受け取ります。
* X (arr): トレーニング セット の特徴。
* y (arr): トレーニング セット のラベル。
  * X_test (arr): テストセット の特徴。
* y_test (arr): テストセット のラベル。

### エルボー プロット

{{< img src="/images/integrations/scikit_elbow_plot.png" alt="" >}}

クラスター数 の関数として説明される分散のパーセンテージ を、トレーニング時間とともに測定してプロット します。最適なクラスター数 を選択するのに役立ちます。

`wandb.sklearn.plot_elbow_curve(model, X_train)`

* model (clusterer): 適合されたクラスタラー を受け取ります。
* X (arr): トレーニング セット の特徴。

### シルエット プロット

{{< img src="/images/integrations/scikit_silhouette_plot.png" alt="" >}}

1 つのクラスター 内の各ポイント が隣接するクラスター 内のポイント にどれだけ近いかを測定してプロット します。クラスター の太さは、クラスター サイズ に対応します。垂直線は、すべてのポイント の平均シルエット スコア を表します。

+1 に近いシルエット係数 は、サンプル が隣接するクラスター から遠く離れていることを示します。0 の値 は、サンプル が 2 つの隣接するクラスター 間の決定境界上にあるか、非常に近いことを示し、負の値 は、それらのサンプル が間違ったクラスター に割り当てられている可能性があることを示します。

一般に、すべてのシルエット クラスター スコア が平均 (赤い線を超える) より高く、できるだけ 1 に近いことが望ましいです。また、データ の基礎となるパターン を反映するクラスター サイズ を推奨します。

`wandb.sklearn.plot_silhouette(model, X_train, ['spam', 'not spam'])`

* model (clusterer): 適合されたクラスタラー を受け取ります。
* X (arr): トレーニング セット の特徴。
  * cluster_labels (list): クラスター ラベル の名前。クラスター インデックス を対応する名前に置き換えることで、プロット を読みやすくします。

### 外れ値候補 プロット

{{< img src="/images/integrations/scikit_outlier_plot.png" alt="" >}}

Cook の距離 を介して、回帰モデル に対するデータポイント の影響を測定します。大きく歪んだ影響を持つインスタンス は、潜在的な外れ値 である可能性があります。外れ値の検出に役立ちます。

`wandb.sklearn.plot_outlier_candidates(model, X, y)`

* model (regressor): 適合された分類器 を受け取ります。
* X (arr): トレーニング セット の特徴。
* y (arr): トレーニング セット のラベル。

### 残差プロット

{{< img src="/images/integrations/scikit_residuals_plot.png" alt="" >}}

予測されたターゲット値 (y 軸) 対 実際のターゲット値 と 予測されたターゲット値 の差 (x 軸)、および残差誤差 の分布を測定してプロット します。

一般に、適切に適合されたモデル の残差 はランダム に分布している必要があります。これは、優れたモデル はランダム誤差 を除いて、データセット 内のほとんどの現象を説明するためです。

`wandb.sklearn.plot_residuals(model, X, y)`

* model (regressor): 適合された分類器 を受け取ります。
* X (arr): トレーニング セット の特徴。
*   y (arr): トレーニング セット のラベル。

    ご不明な点がございましたら、[Slack コミュニティ](http://wandb.me/slack) でお気軽にお問い合わせください。

## 例

* [colab で実行](http://wandb.me/scikit-colab): 開始するための簡単な ノートブック
