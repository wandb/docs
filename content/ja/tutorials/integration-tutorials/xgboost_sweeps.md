---
title: XGBoost Sweeps
menu:
  tutorials:
    identifier: ja-tutorials-integration-tutorials-xgboost_sweeps
    parent: integration-tutorials
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Using_W&B_Sweeps_with_XGBoost.ipynb" >}}
Weights & Biases を使用して、機械学習の実験管理、データセットのバージョン管理、プロジェクトコラボレーションを行いましょう。

{{< img src="/images/tutorials/huggingface-why.png" alt="" >}}

ツリーベースのモデルから最高のパフォーマンスを引き出すためには、[正しいハイパーパラメーターを選択する](https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f)ことが必要です。いくつの `early_stopping_rounds` が必要か？ツリーの `max_depth` は何にすべきか？

高次元のハイパーパラメータースペースを探索して、最もパフォーマンスの良いモデルを見つけることは、非常に煩雑になる可能性があります。ハイパーパラメーター探索は、モデルのバトルロワイヤルを組織的かつ効率的に行い、勝者を決める方法を提供します。これにより、ハイパーパラメーターの値の組み合わせを自動的に探索し、最適な値を見つけることができるのです。

このチュートリアルでは、Weights & Biases を使用して、XGBoost モデルに対して高度なハイパーパラメーター探索を3つの簡単なステップで実行する方法を見ていきます。

まず、以下のプロットをチェックしてみてください:

{{< img src="/images/tutorials/xgboost_sweeps/sweeps_xgboost.png" alt="sweeps_xgboost" >}}

## Sweeps: An Overview

Weights & Biases でハイパーパラメーター探索を行うのは非常に簡単です。3つのシンプルなステップだけです:

1. **スイープを定義する:** 探索するパラメーター、使用する探索戦略、最適化するメトリクスを指定する辞書のようなオブジェクトを作成して、スイープを定義します。

2. **スイープを初期化する:** 1行のコードでスイープを初期化し、スイープ設定の辞書を渡します:
`sweep_id = wandb.sweep(sweep_config)`

3. **スイープエージェントを実行する:** これも1行のコードで達成され、`wandb.agent()` を呼び出し、`sweep_id` とモデルアーキテクチャーを定義しトレーニングを行う関数を渡します:
`wandb.agent(sweep_id, function=train)`

これでハイパーパラメーター探索を実行する準備が整いました。

以下のノートブックでは、これらの3つのステップを詳しく見ていきます。

このノートブックをフォークして、パラメーターを調整したり、ご自身のデータセットでモデルを試してみることを強くお勧めします。

### リソース
- [Sweeps docs →]({{< relref path="/guides/models/sweeps/" lang="ja" >}})
- [コマンドラインからのローンチ →](https://www.wandb.com/articles/hyperparameter-tuning-as-easy-as-1-2-3)



```python
!pip install wandb -qU
```


```python

import wandb
wandb.login()
```

## 1. スイープを定義する

Weights & Biases の sweeps は、わずか数行のコードで、sweeps を思い通りに設定するための強力なレバーを提供します。sweeps の設定は、[辞書や YAML ファイル]({{< relref path="/guides/models/sweeps/define-sweep-configuration" lang="ja" >}})として定義できます。

一緒にいくつか見ていきましょう:
*   **メトリクス**: これは sweeps が最適化しようとしているメトリクスです。メトリクスには`name`（このメトリクスはトレーニングスクリプトでログインされている必要があります）と `goal`（`maximize` または `minimize`）を指定できます。
*   **探索戦略**: `"method"` キーを使用して指定します。sweeps ではいくつかの異なる探索戦略をサポートしています。
  *   **グリッド検索**: ハイパーパラメーター値のすべての組み合わせを反復します。
  *   **ランダム検索**: ランダムに選ばれたハイパーパラメーター値の組み合わせを反復します。
  *   **ベイズ検索**: ハイパーパラメーターをメトリックスコアの確率にマッピングする確率モデルを作成し、メトリクスの改善の可能性が高いパラメータを選択します。ベイズ最適化の目的は、ハイパーパラメーター値を選ぶことにより多くの時間を費やしつつ、より少ないハイパーパラメーター値を試すことです。
*   **パラメーター**: ハイパーパラメーター名と、それぞれのイテレーションでの値を取得する範囲や分布を含む辞書です。

詳細については、[すべてのスイープ設定オプションのリスト]({{< relref path="/guides/models/sweeps/define-sweep-configuration" lang="ja" >}})を参照してください。


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

## 2. スイープを初期化する

`wandb.sweep` を呼び出すと、スイープコントローラが開始されます --
これは設定された `parameters` をクエリする人々に設定を提供し、
`wandb` ログを通じて `metrics` のパフォーマンスを返すことを期待します。


```python
sweep_id = wandb.sweep(sweep_config, project="XGBoost-sweeps")
```

### トレーニングプロセスを定義する
スイープを実行する前に、
モデルを作成しトレーニングする関数を定義する必要があります --
ハイパーパラメーターの値を受け取り、メトリクスを出力する関数です。

また、`wandb` をスクリプトに組み込む必要があります。
主なコンポーネントは3つです:
*   `wandb.init()`: 新しい W&B run を初期化します。各 run はトレーニングスクリプトの単一の実行です。
*   `wandb.config`: すべてのハイパーパラメーターを設定オブジェクトに保存します。これにより、[アプリ](https://wandb.ai)を使用してハイパーパラメーターの値で run をソートし比較することができます。
*   `wandb.log()`: メトリクスや画像、動画、音声ファイル、HTML、プロット、ポイントクラウドなどのカスタムオブジェクトをログします。

また、データをダウンロードする必要があります:


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

## 3. エージェントを使ってスイープを実行する

さて、`wandb.agent` を呼び出してスイープを開始しましょう。

`wandb.agent` を以下の条件を満たす W&B にログインした任意のマシンで呼び出すことができます。
- `sweep_id`
- データセットと `train` 関数

そして、そのマシンはスイープに参加します。

> _注意_: `random` スイープはデフォルトで永遠に実行され、
新しいパラメーターの組み合わせを試し続けます --
それを[アプリの UI からスイープをオフにするまで]({{< relref path="/guides/models/sweeps/sweeps-ui" lang="ja" >}})。
`agent` に完了する run の合計 `count` を指定することでこれを防ぐことができます。


```python
wandb.agent(sweep_id, train, count=25)
```

## 結果を可視化する

スイープが終了したら、結果を確認する時が来ました。

Weights & Biases は自動的に多数の便利なプロットを生成します。

### Parallel coordinates plot

このプロットはハイパーパラメーターの値をモデルのメトリクスにマッピングします。このプロットは最も良いモデルパフォーマンスを導いたハイパーパラメーターの組み合わせを特定するのに役立ちます。

このプロットは、学習者としてツリーを使用することがわずかに優れていることを示しているようです。
しかし、圧倒的ではありません。
シンプルな線形モデルを学習者として使用するよりもやや優れています。

{{< img src="/images/tutorials/xgboost_sweeps/sweeps_xgboost2.png" alt="sweeps_xgboost" >}}

### ハイパーパラメーターの重要性プロット

ハイパーパラメーターの重要性プロットは、どのハイパーパラメーターの値があなたのメトリクスに最も大きな影響を与えたかを示します。

相関（線形予測子として扱っている）と特徴の重要性（結果に基づいてランダムフォレストをトレーニングした後）を報告し、
どのパラメーターが最も大きな影響を与えたか、
そしてその影響が正か負かがわかるようになっています。

このチャートを読むことにより、
上記の並列座標チャートで気づいた傾向の定量的な確認を見ることができます：
検証精度に対する最大の影響は
学習者の選択から来ており、
`gblinear` 学習者は一般的に `gbtree` 学習者よりも悪いことがわかります。

{{< img src="/images/tutorials/xgboost_sweeps/sweeps_xgboost3.png" alt="sweeps_xgboost" >}}

これらの可視化は、最も重要でさらに探求する価値のあるパラメーター（とその値の範囲）に焦点を当てることにより、
高価なハイパーパラメーターの最適化の実行にかかる時間とリソースを節約するのに役立ちます。