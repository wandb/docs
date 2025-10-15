---
title: XGBoost Sweeps
menu:
  tutorials:
    identifier: ja-tutorials-integration-tutorials-xgboost_sweeps
    parent: integration-tutorials
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Using_W&B_Sweeps_with_XGBoost.ipynb" >}}
Weights & Biases を使って機械学習の実験管理、データセットのバージョン管理、プロジェクトの協力を行いましょう。

{{< img src="/images/tutorials/huggingface-why.png" alt="" >}}

ツリー型モデルから最良のパフォーマンスを引き出すには、[適切なハイパーパラメーターを選択](https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f)する必要があります。いくつの `early_stopping_rounds` が必要でしょうか？ ツリーの `max_depth` はどのくらいにすべきですか？

高次元のハイパーパラメータ空間を検索して最適なモデルを見つけることは非常に困難です。ハイパーパラメーター探索は、モデルのバトルロイヤルを組織的かつ効率的に実施し、勝者を決定する方法を提供します。これにより、ハイパーパラメータの組み合わせを自動的に検索して、最も最適な値を見つけ出すことができます。

このチュートリアルでは、Weights & Biases を使用して XGBoost モデルで洗練されたハイパーパラメーター探索を 3 つの簡単なステップで実行する方法を紹介します。

興味を引くために、以下のプロットをチェックしてください：

{{< img src="/images/tutorials/xgboost_sweeps/sweeps_xgboost.png" alt="sweeps_xgboost" >}}

## Sweeps: 概要

Weights & Biases を使ったハイパーパラメーター探索の実行は非常に簡単です。以下の3つのシンプルなステップです：

1. **スイープを定義する:** スイープを定義するためには、スイープを構成するパラメータ、使用する検索戦略、最適化するメトリクスを指定する辞書のようなオブジェクトを作成します。

2. **スイープを初期化する:** 1 行のコードでスイープを初期化し、スイープの設定の辞書を渡します：
`sweep_id = wandb.sweep(sweep_config)`

3. **スイープエージェントを実行する:** これも 1 行のコードで、`wandb.agent()` を呼び出し、`sweep_id` とモデルアーキテクチャを定義してトレーニングする関数を渡します：
`wandb.agent(sweep_id, function=train)`

これだけでハイパーパラメーター探索を実行することができます。

以下のノートブックでは、これらの 3 ステップを詳細に説明します。

ぜひこのノートブックをフォークして、パラメータを調整したり、独自のデータセットでモデルを試してみてください。

### リソース
- [Sweeps のドキュメント →]({{< relref path="/guides/models/sweeps/" lang="ja" >}})
- [コマンドラインからの起動 →](https://www.wandb.com/articles/hyperparameter-tuning-as-easy-as-1-2-3)



```python
!pip install wandb -qU
```


```python

import wandb
wandb.login()
```

## 1. スイープを定義する

Weights & Biases の Sweeps は、希望通りにスイープを設定するための強力なレバーを少ないコード行で提供します。スイープの設定は、[辞書または YAML ファイル]({{< relref path="/guides/models/sweeps/define-sweep-configuration" lang="ja" >}})として定義できます。

それらのいくつかを一緒に見ていきましょう：
*   **メトリック**: これは、スイープが最適化しようとしているメトリックです。メトリクスは `name` (トレーニングスクリプトでログされるべきメトリック名) と `goal` (`maximize` か `minimize`) を取ることができます。
*   **検索戦略**: `"method"` キーを使用して指定されます。スイープでは、いくつかの異なる検索戦略をサポートしています。
  *   **グリッド検索**: ハイパーパラメーター値のすべての組み合わせを反復します。
  *   **ランダム検索**: ランダムに選ばれたハイパーパラメーター値の組み合わせを反復します。
  *   **ベイズ探索**: ハイパーパラメーターをメトリクススコアの確率とマッピングする確率モデルを作成し、メトリクスを改善する高い確率のパラメータを選択します。ベイズ最適化の目的は、ハイパーパラメーター値の選択に時間をかけることですが、その代わりにより少ないハイパーパラメーター値を試すことを試みます。
*   **パラメータ**: ハイパーパラメータ名、離散値、範囲、または各反復で値を取り出す分布を含む辞書です。

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

`wandb.sweep` を呼び出すとスイープコントローラー、つまり `parameters` の設定を問い合わせるすべての者に提供し、`metrics` のパフォーマンスを `wandb` ログを介して返すことを期待する集中プロセスが開始されます。


```python
sweep_id = wandb.sweep(sweep_config, project="XGBoost-sweeps")
```

### トレーニングプロセスを定義する
スイープを実行する前に、モデルを作成してトレーニングする関数を定義する必要があります。
この関数は、ハイパーパラメーター値を取り込み、メトリクスを出力するものです。

また、スクリプト内に `wandb` を統合する必要があります。
主なコンポーネントは3つです：
*   `wandb.init()`: 新しい W&B run を初期化します。各 run はトレーニングスクリプトの単一の実行です。
*   `wandb.config`: すべてのハイパーパラメーターを設定 オブジェクトに保存します。これにより、[私たちのアプリ](https://wandb.ai)を使用して、ハイパーパラメーター値ごとに run をソートおよび比較することができます。
*   `wandb.log()`: 画像、ビデオ、オーディオファイル、HTML、プロット、またはポイントクラウドなどのメトリクスとカスタムオブジェクトをログします。

また、データをダウンロードする必要があります：


```python
!wget https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
```


```python
# Pima Indians データセット用の XGBoost モデル
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# データを読み込む
def train():
  config_defaults = {
    "booster": "gbtree",
    "max_depth": 3,
    "learning_rate": 0.1,
    "subsample": 1,
    "seed": 117,
    "test_size": 0.33,
  }

  wandb.init(config=config_defaults)  # スイープ中にデフォルトが上書きされる
  config = wandb.config

  # データを読み込み、予測変数とターゲットに分ける
  dataset = loadtxt("pima-indians-diabetes.data.csv", delimiter=",")
  X, Y = dataset[:, :8], dataset[:, 8]

  # データをトレインセットとテストセットに分割
  X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                      test_size=config.test_size,
                                                      random_state=config.seed)

  # トレインセットでモデルを適合させる
  model = XGBClassifier(booster=config.booster, max_depth=config.max_depth,
                        learning_rate=config.learning_rate, subsample=config.subsample)
  model.fit(X_train, y_train)

  # テストセットで予測を行う
  y_pred = model.predict(X_test)
  predictions = [round(value) for value in y_pred]

  # 予測を評価する
  accuracy = accuracy_score(y_test, predictions)
  print(f"Accuracy: {accuracy:.0%}")
  wandb.log({"accuracy": accuracy})
```

## 3. エージェントでスイープを実行する

次に、`wandb.agent` を呼び出してスイープを起動します。

`wandb.agent` は W&B にログインしているすべてのマシンで呼び出すことができ、
- `sweep_id`
- データセットと `train` 関数

があるので、そのマシンはスイープに参加します。

> _注意_: `random` スイープはデフォルトでは永遠に実行され、新しいパラメータの組み合わせを試し続けます。しかし、それは[アプリの UI からスイープをオフにするまで]({{< relref path="/guides/models/sweeps/sweeps-ui" lang="ja" >}})です。完了する run の合計 `count` を `agent` に指定することで、この動作を防ぐことができます。


```python
wandb.agent(sweep_id, train, count=25)
```

## 結果を可視化する

スイープが終了したら、結果を確認します。

Weights & Biases は、あなたのために便利なプロットをいくつか自動的に生成します。

### パラレル座標プロット

このプロットは、ハイパーパラメーター値をモデルのメトリクスにマッピングします。最良のモデルパフォーマンスをもたらしたハイパーパラメーターの組み合わせに焦点を当てるのに役立ちます。

このプロットは、単純な線形モデルを学習者として使用するよりも、ツリーを学習者として使用する方がやや優れていることを示しているようです。

{{< img src="/images/tutorials/xgboost_sweeps/sweeps_xgboost2.png" alt="sweeps_xgboost" >}}

### ハイパーパラメーターの重要度プロット

ハイパーパラメーターの重要度プロットは、メトリクスに最も大きな影響を与えたハイパーパラメーター値を示しています。

相関関係（線形予測子として扱う）と特徴量の重要性（結果に基づいてランダムフォレストをトレーニングした後） の両方を報告しますので、どのパラメータが最も大きな影響を与えたか、そしてその影響がどちらの方向であったかを確認できます。

このチャートを読むと、上記のパラレル座標チャートで気づいた傾向の定量的な確認が得られます。検証精度に最大の影響を与えたのは学習者の選択であり、`gblinear` 学習者は一般的に `gbtree` 学習者よりも劣っていました。

{{< img src="/images/tutorials/xgboost_sweeps/sweeps_xgboost3.png" alt="sweeps_xgboost" >}}

これらの可視化は、最も重要で、さらに探索する価値のあるパラメータ（とその値の範囲）に焦点を当てることにより、高価なハイパーパラメーターの最適化を実行する時間とリソースを節約するのに役立ちます。