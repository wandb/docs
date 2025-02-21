---
title: XGBoost Sweeps
menu:
  tutorials:
    identifier: ja-tutorials-integration-tutorials-xgboost_sweeps
    parent: integration-tutorials
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Using_W&B_Sweeps_with_XGBoost.ipynb" >}}
Weights & Biases を使用して、 機械学習 の 実験管理 、 データセット の バージョン管理 、 プロジェクト の コラボレーションを実現しましょう。

{{< img src="/images/tutorials/huggingface-why.png" alt="" >}}

ツリー ベースの モデル の パフォーマンス を最大限に引き出すには、[適切な ハイパーパラメーター を選択する](https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f)必要があります。`early_stopping_rounds` はいくつにすべきでしょうか？ツリー の `max_depth` はどのくらいにすべきでしょうか？

高次元 の ハイパーパラメーター 空間 を探索して、最もパフォーマンス の 高い モデル を見つけ出すのは、非常に困難になる可能性があります。 ハイパーパラメーター Sweeps は、 モデル の 総当たり戦 を行い、勝者を決定するための組織的かつ効率的な方法を提供します。これは、ハイパーパラメーター 値 の 組み合わせ を自動的に検索して、最適な 値 を見つけることによって実現されます。

この チュートリアル では、Weights & Biases を使用して、3 つ の 簡単な ステップ で XGBoost モデル で高度な ハイパーパラメーター Sweeps を実行する方法を説明します。

まず、以下 の プロット をご覧ください。

{{< img src="/images/tutorials/xgboost_sweeps/sweeps_xgboost.png" alt="sweeps_xgboost" >}}

## Sweeps：概要

Weights & Biases で ハイパーパラメーター sweep を実行する の は非常に簡単です。簡単な 3 つ の ステップ があります。

1.  **sweep を定義する**: sweep を指定する 辞書 の よう な オブジェクト を作成して、これを行います。検索する パラメータ ー、使用する 検索 戦略 、最適化する メトリクス を指定します。

2.  **sweep を初期化する**: 1 行 の コード で sweep を初期化し、sweep 設定 の 辞書 を渡します。
    `sweep_id = wandb.sweep(sweep_config)`

3.  **sweep agent を実行する**: これも 1 行 の コード で実行できます。w`andb.agent()` を呼び出し、`sweep_id` と モデル アーキテクチャー を定義して トレーニング する 関数 を渡します。
    `wandb.agent(sweep_id, function=train)`

以上で ハイパーパラメーター sweep の実行は完了です。

以下 の ノートブック では、これら 3 つ の ステップ について詳しく説明します。

この ノートブック を フォーク して、 パラメータ ー を調整したり、独自の データセット で モデル を試したりすることを強くお勧めします。

### リソース
- [Sweeps の ドキュメント →]({{< relref path="/guides/models/sweeps/" lang="ja" >}})
- [コマンドライン から の 起動 →](https://www.wandb.com/articles/hyperparameter-tuning-as-easy-as-1-2-3)

```python
!pip install wandb -qU
```

```python
import wandb
wandb.login()
```

## 1. Sweep を定義する

Weights & Biases Sweeps を使用すると、わずか数行 の コード で、sweep を正確に構成するための強力な レバー を利用できます。Sweeps の 設定 は、[辞書 または YAML ファイル]({{< relref path="/guides/models/sweeps/define-sweep-configuration" lang="ja" >}})として定義できます。

いくつか一緒に見ていきましょう。
*   **メトリクス**: これは、Sweeps が最適化しようとしている メトリクス です。メトリクス は、`name`（この メトリクス は トレーニング スクリプト によって ログ に記録される必要があります）と `goal`（`maximize` または `minimize`）を受け取ることができます。
*   **検索 戦略**: `"method"` キー を使用して指定します。Sweeps では、いくつか の 異なる 検索 戦略 がサポートされています。
    *   **グリッド検索**: ハイパーパラメーター 値 の すべて の 組み合わせ を反復処理します。
    *   **ランダム検索**: ランダム に選択された ハイパーパラメーター 値 の 組み合わせ を反復処理します。
    *   **ベイズ探索**: ハイパーパラメーター を メトリクス スコア の 確率 に マッピング する 確率 モデル を作成し、 メトリクス を改善する 確率 が高い パラメータ ー を選択します。 ベイズ最適化 の 目的 は、ハイパーパラメーター 値 の 選択 により多くの時間を費やすことですが、そうすることで、より少ない ハイパーパラメーター 値 を試すことです。
*   **パラメータ ー**: ハイパーパラメーター 名、 離散値 、範囲、または 各反復 で 値 を取得する 分布 を含む 辞書 。

詳細については、[すべて の sweep 設定 オプション の リスト]({{< relref path="/guides/models/sweeps/define-sweep-configuration" lang="ja" >}})を参照してください。

```python
sweep_config = {
    "method": "random", # try grid or random
    "metric": {
      "name": "accuracy",
      "goal": "maximize"   
    },
    "parameters": {
        "booster": {
            "values": ["gbtree","gblinear"]
        },
        "max_depth": {
            "values": [3, 6, 9, 12]
        },
        "learning_rate": {
            "values": [0.1, 0.05, 0.2]
        },
        "subsample": {
            "values": [1, 0.5, 0.3]
        }
    }
}
```

## 2. Sweep を初期化する

`wandb.sweep` を呼び出すと、Sweep Controller が開始されます。これは、`parameters` の 設定 を クエリ する すべて の ユーザー に提供し、`wandb` ログ を介して `metrics` の パフォーマンス を返すことを期待する集中型 プロセス です。

```python
sweep_id = wandb.sweep(sweep_config, project="XGBoost-sweeps")
```

### トレーニング プロセス を定義する
sweep を実行する 前 に、 モデル を作成および トレーニング する 関数 を定義する必要があります。これは、ハイパーパラメーター 値 を受け取り、 メトリクス を出力する 関数 です。

また、`wandb` が スクリプト に 統合 されている必要もあります。主な コンポーネント は 3 つ あります。
*   `wandb.init()`: 新しい W&B run を初期化します。各 run は、 トレーニング スクリプト の 1 回 の 実行です。
*   `wandb.config`: すべて の ハイパーパラメーター を config オブジェクト に保存します。これにより、[アプリ](https://wandb.ai) を使用して、ハイパーパラメーター 値 で run を並べ替えたり、比較したりできます。
*   `wandb.log()`: メトリクス や、画像、動画、 音声ファイル 、HTML、 プロット 、 ポイントクラウド など の カスタム オブジェクト を ログ に記録します。

また、 データ をダウンロードする必要もあります。

```python
!wget https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
```

```python
# XGBoost model for Pima Indians dataset
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load data
def train():
  config_defaults = {
    "booster": "gbtree",
    "max_depth": 3,
    "learning_rate": 0.1,
    "subsample": 1,
    "seed": 117,
    "test_size": 0.33,
  }

  wandb.init(config=config_defaults)  # defaults are over-ridden during the sweep
  config = wandb.config

  # load data and split into predictors and targets
  dataset = loadtxt("pima-indians-diabetes.data.csv", delimiter=",")
  X, Y = dataset[:, :8], dataset[:, 8]

  # split data into train and test sets
  X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                      test_size=config.test_size,
                                                      random_state=config.seed)

  # fit model on train
  model = XGBClassifier(booster=config.booster, max_depth=config.max_depth,
                        learning_rate=config.learning_rate, subsample=config.subsample)
  model.fit(X_train, y_train)

  # make predictions on test
  y_pred = model.predict(X_test)
  predictions = [round(value) for value in y_pred]

  # evaluate predictions
  accuracy = accuracy_score(y_test, predictions)
  print(f"Accuracy: {accuracy:.0%}")
  wandb.log({"accuracy": accuracy})
```

## 3. agent で Sweep を実行する

次に、`wandb.agent` を呼び出して sweep を起動します。

W&B に ログイン している マシン で `wandb.agent` を呼び出すことができます。
- `sweep_id`、
- データセット と `train` 関数

その マシン は sweep に参加します。

> _注_: `random` sweep は、デフォルト で 無期限 に実行され、新しい パラメータ ー の 組み合わせ を試します。[アプリ UI から sweep をオフにする]({{< relref path="/guides/models/sweeps/sweeps-ui" lang="ja" >}})まで。`agent` に完了させたい run の 合計 `count` を指定することで、これを防ぐことができます。

```python
wandb.agent(sweep_id, train, count=25)
```

## 結果 を 可視化する

sweep が完了したので、結果 を見てみましょう。

Weights & Biases は、多くの役立つ プロット を自動的に生成します。

### 並列座標 プロット

この プロット は、ハイパーパラメーター 値 を モデル メトリクス に マッピング します。これは、最高 の モデル パフォーマンス につながった ハイパーパラメーター の 組み合わせ を絞り込むのに役立ちます。

この プロット は、学習者として ツリー を使用すると、単純な 線形 モデル を学習者として使用するよりもわずかに優れていることを示しているようですが、驚くほどではありません。

{{< img src="/images/tutorials/xgboost_sweeps/sweeps_xgboost2.png" alt="sweeps_xgboost" >}}

### ハイパーパラメーター の 重要性 プロット

ハイパーパラメーター の 重要性 プロット は、どの ハイパーパラメーター 値 が メトリクス に 最大 の 影響 を与えたかを示しています。

相関関係（ 線形 予測子 として扱う）と 特徴 の 重要性 （結果 に対して ランダムフォレスト を トレーニング した後） の 両方 を報告するため、どの パラメータ ー が 最大 の 影響 を与えたか、および その 影響 が プラス か マイナス かを 確認できます。

この チャート を読むと、上記 の 並列座標 チャート で 気付いた 傾向 の 定量的な 確認 が得られます。 検証精度 に 最大 の 影響 を与えた の は、学習者 の 選択 であり、`gblinear` 学習者 は 一般に `gbtree` 学習者 よりも劣っていました。

{{< img src="/images/tutorials/xgboost_sweeps/sweeps_xgboost3.png" alt="sweeps_xgboost" >}}

これら の 可視化 は、最も重要で、それによって さらに 調査する価値 がある パラメータ ー (および 値 の 範囲) を絞り込むことで、高価な ハイパーパラメーター 最適化 を実行する 時間 と リソース の 両方 を節約するのに役立ちます。
