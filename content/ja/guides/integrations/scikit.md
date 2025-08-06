---
title: Scikit-Learn
menu:
  default:
    identifier: scikit
    parent: integrations
weight: 380
---

wandb を使えば、わずか数行のコードで scikit-learn モデルのパフォーマンスを可視化し、比較できます。[サンプルを試す →](https://wandb.me/scikit-colab)

## はじめに

### サインアップと APIキー の作成

APIキー は、あなたのマシンを W&B に認証するために使われます。APIキー はユーザープロフィールから生成できます。

{{% alert %}}
より簡単な方法として、[W&B の認証ページ](https://wandb.ai/authorize)から直接 APIキー を生成できます。表示された APIキー をコピーして、パスワードマネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上にあるご自身のユーザーアイコンをクリックします。
1. **User Settings** を選択し、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックします。表示された APIキー をコピーしてください。APIキー を非表示にするには、ページを再読み込みします。

### `wandb` ライブラリをインストールしてログイン

ローカルに `wandb` ライブラリをインストールし、ログインします:

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref "/guides/models/track/environment-variables.md" >}}) に APIキー を設定します。

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

  # メトリクスを時系列で記録する場合は run.log を使います
  run.log({"accuracy": accuracy})

  # または、トレーニングが終了した後に最終的なメトリクスだけを記録したい場合は run.summary も使えます
  run.summary["accuracy"] = accuracy
```

### グラフの作成

#### ステップ 1: wandb をインポートし、新しい run を初期化

```python
import wandb

run = wandb.init(project="visualize-sklearn")
```

#### ステップ 2: グラフを可視化

#### 個別プロット

モデルをトレーニングし、予測した後に、wandb でグラフを生成し予測を分析できます。サポートされているチャートの一覧は下記の **Supported Plots** セクションをご覧ください。

```python
# 単一のプロットを可視化
wandb.sklearn.plot_confusion_matrix(y_true, y_pred, labels)
```

#### すべてのプロット

W&B には `plot_classifier` など、関連する複数のグラフを一度に作成する関数があります:

```python
# すべての分類プロットを可視化
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

# 回帰のグラフすべて
wandb.sklearn.plot_regressor(reg, X_train, X_test, y_train, y_test, model_name="Ridge")

# クラスタリングのグラフすべて
wandb.sklearn.plot_clusterer(
    kmeans, X_train, cluster_labels, labels=None, model_name="KMeans"
)

run.finish()
```

#### 既存の Matplotlib グラフ

Matplotlib で作成したグラフも W&B のダッシュボードにログできます。まず最初に、`plotly` のインストールが必要です。

```bash
pip install plotly
```

最後に、次のように W&B のダッシュボードにグラフを記録できます:

```python
import matplotlib.pyplot as plt
import wandb

with wandb.init(project="visualize-sklearn") as run:

  # ここで plt.plot(), plt.scatter() などを実行
  # ...

  # plt.show() の代わりに以下を実行
  run.log({"plot": plt})
```

## サポートされているプロット

### 学習曲線

{{< img src="/images/integrations/scikit_learning_curve.png" alt="Scikit-learn learning curve" >}}

さまざまな長さのデータセットでモデルを学習し、交差検証スコアとデータセットサイズのプロットを作成します（トレーニングセット・テストセット両方）。

`wandb.sklearn.plot_learning_curve(model, X, y)`

* model (clf または reg): 学習済み回帰器または分類器
* X (arr): データセットの特徴量
* y (arr): データセットのラベル

### ROC

{{< img src="/images/integrations/scikit_roc.png" alt="Scikit-learn ROC curve" >}}

ROC曲線は、真陽性率（y軸）と偽陽性率（x軸）をプロットします。理想的なスコアは TPR=1, FPR=0、グラフの左上です。通常、ROC曲線下面積（AUC-ROC）を計算し、AUCが大きいほど性能が良いとされます。

`wandb.sklearn.plot_roc(y_true, y_probas, labels)`

* y_true (arr): テストセットのラベル
* y_probas (arr): テストセットの推定確率
* labels (list): 目的変数 (y) の名前付きラベル

### クラス分布

{{< img src="/images/integrations/scikic_class_props.png" alt="Scikit-learn classification properties" >}}

トレーニングセットおよびテストセット内の各クラスの分布をプロットします。クラスの不均衡や、特定のクラスがモデルへの影響で過大になっていないかの確認に有用です。

`wandb.sklearn.plot_class_proportions(y_train, y_test, ['dog', 'cat', 'owl'])`

* y_train (arr): トレーニングセットのラベル
* y_test (arr): テストセットのラベル
* labels (list): 目的変数 (y) の名前付きラベル

### Precision-Recall カーブ

{{< img src="/images/integrations/scikit_precision_recall.png" alt="Scikit-learn precision-recall curve" >}}

異なる閾値での Precision と Recall のトレードオフを計算します。曲線下面積が大きいほど、Precision も Recall も高く、誤認率・取りこぼし率の両方が低いことを示します。

どちらも高い場合、分類器が正確な結果（高い Precision）を返すと同時に、大部分の正例も検出できている（高い Recall）ことになります。PR曲線は、クラスが大きく不均衡な場合に有用です。

`wandb.sklearn.plot_precision_recall(y_true, y_probas, labels)`

* y_true (arr): テストセットのラベル
* y_probas (arr): テストセットの推定確率
* labels (list): 目的変数 (y) の名前付きラベル

### 特徴量重要度

{{< img src="/images/integrations/scikit_feature_importances.png" alt="Scikit-learn feature importance chart" >}}

各特徴量の分類タスクへの寄与度を評価・プロットします。`feature_importances_` 属性を持つ分類器（ツリー系など）で利用できます。

`wandb.sklearn.plot_feature_importances(model, ['width', 'height', 'length'])`

* model (clf): 学習済み分類器
* feature_names (list): 特徴量の名前。インデックスの代わりに分かりやすい名前が表示され、グラフの可読性向上に役立ちます。

### キャリブレーションカーブ

{{< img src="/images/integrations/scikit_calibration_curve.png" alt="Scikit-learn calibration curve" >}}

分類器が出力した確率のキャリブレーション状況や、非キャリブレーション分類器の調整方法を可視化します。ベースラインロジスティック回帰、引数で渡したモデル、およびそのアイソトニックキャリブレーション・シグモイドキャリブレーションを比較します。

キャリブレーションカーブが対角線に近いほどよいです。シグモイド型が反転した曲線はモデルの過学習、通常のシグモイド型は学習不足を示します。アイソトニックとシグモイドの調整結果を比較することで、モデルが過学習か学習不足かを推測し、必要に応じて適切なキャリブレーションを検討できます。

詳細は [sklearn のドキュメント](https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html) をご参照ください。

`wandb.sklearn.plot_calibration_curve(clf, X, y, 'RandomForestClassifier')`

* model (clf): 学習済み分類器
* X (arr): トレーニングセット特徴量
* y (arr): トレーニングセットラベル
* model_name (str): モデル名（デフォルトは 'Classifier'）

### 混同行列

{{< img src="/images/integrations/scikit_confusion_matrix.png" alt="Scikit-learn confusion matrix" >}}

分類の正確性を評価するために混同行列を計算します。モデルの予測精度や、誤分類が発生しやすいパターンの把握に役立ちます。対角要素は、実際のラベルと予測ラベルが一致した数を示しています。

`wandb.sklearn.plot_confusion_matrix(y_true, y_pred, labels)`

* y_true (arr): テストセットのラベル
* y_pred (arr): テストセットの予測ラベル
* labels (list): 目的変数 (y) の名前付きラベル

### サマリーメトリクス

{{< img src="/images/integrations/scikit_summary_metrics.png" alt="Scikit-learn summary metrics" >}}

- 分類では `mse`、`mae`、`r2` スコアなどのサマリーメトリクスを計算します。
- 回帰では `f1`、accuracy、precision、recall などのサマリーメトリクスを計算します。

`wandb.sklearn.plot_summary_metrics(model, X_train, y_train, X_test, y_test)`

* model (clf または reg): 学習済み回帰器または分類器
* X (arr): トレーニングセット特徴量
* y (arr): トレーニングセットラベル
  * X_test (arr): テストセット特徴量
* y_test (arr): テストセットラベル

### エルボープロット

{{< img src="/images/integrations/scikit_elbow_plot.png" alt="Scikit-learn elbow plot" >}}

クラスタ数に対する分散説明率とトレーニング時間をプロットします。最適なクラスタ数選択の参考になります。

`wandb.sklearn.plot_elbow_curve(model, X_train)`

* model (clusterer): 学習済みクラスタラー
* X (arr): トレーニングセット特徴量

### シルエットプロット

{{< img src="/images/integrations/scikit_silhouette_plot.png" alt="Scikit-learn silhouette plot" >}}

各クラスタにあるデータ点が隣接クラスタの点からどれだけ離れているかをプロットします。クラスターの厚みはサイズに対応し、縦の線はクラスター全体の平均シルエットスコアを示します。

シルエット係数が+1に近いほど、サンプルが他のクラスタから十分離れていることを意味します。値が0は2つのクラスタの境界上、負の値は誤ったクラスター割り当ての可能性を示します。

全体的に、すべてのシルエットスコアが平均以上（赤線より右）で、1に近いことが望ましいです。また、クラスターのサイズはデータに内在するパターンを反映している方が良いです。

`wandb.sklearn.plot_silhouette(model, X_train, ['spam', 'not spam'])`

* model (clusterer): 学習済みクラスタラー
* X (arr): トレーニングセット特徴量
  * cluster_labels (list): クラスターラベルの名前。インデックスではなく分かりやすい名前が表示され、グラフの可読性向上に役立ちます。

### 外れ値候補プロット

{{< img src="/images/integrations/scikit_outlier_plot.png" alt="Scikit-learn outlier plot" >}}

回帰モデルに対する各データ点のクック距離を計算し、グラフ化します。影響の大きく偏ったインスタンスは外れ値の可能性があります。外れ値検出に有用です。

`wandb.sklearn.plot_outlier_candidates(model, X, y)`

* model (regressor): 学習済み回帰器
* X (arr): トレーニングセット特徴量
* y (arr): トレーニングセットラベル

### 残差プロット

{{< img src="/images/integrations/scikit_residuals_plot.png" alt="Scikit-learn residuals plot" >}}

予測値（y軸）と実際値と予測値の差分（x軸）、および残差誤差の分布をプロットします。

良いモデルであれば残差はランダムに分布するはずです。これは、データセット内のあらゆる現象をモデルが十分捉え、誤差はランダムのみとなっているためです。

`wandb.sklearn.plot_residuals(model, X, y)`

* model (regressor): 学習済み回帰器
* X (arr): トレーニングセット特徴量
*   y (arr): トレーニングセットラベル

何か質問があれば、[slack コミュニティ](https://wandb.me/slack) でお気軽にどうぞ。

## サンプル

* [colab で実行](https://wandb.me/scikit-colab): すぐに始められるシンプルなノートブックです。