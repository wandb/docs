


# XGBoost Sweeps

[**Colabノートブックで試してみる →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Using_W&B_Sweeps_with_XGBoost.ipynb)

Weights & Biasesを使って機械学習の実験トラッキング、データセットのバージョン管理、およびプロジェクトのコラボレーションを行いましょう。

<img src="http://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />

ツリーベースのモデルから最高のパフォーマンスを引き出すには[適切なハイパーパラメーターの選択](https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f)が必要です。`early_stopping_rounds`は何回にすべきか？ツリーの`max_depth`はどれくらいが良いのか？

高次元のハイパーパラメーター空間を探索して最も性能の良いモデルを見つけるのは非常に大変です。ハイパーパラメータースイープは、モデルのバトルロイヤルを組織的かつ効率的に行い、勝者を見つける手段です。これにより、ハイパーパラメーター値の組み合わせを自動的に検索して、最適な値を見つけることができます。

このチュートリアルでは、Weights & Biasesを用いて、XGBoostモデルで高度なハイパーパラメータースイープを3つの簡単なステップで実行する方法を紹介します。

まずは以下のプロットを見てみましょう：

![sweeps_xgboost](/images/tutorials/xgboost_sweeps/sweeps_xgboost.png)

## Sweeps: 概要

Weights & Biasesでハイパーパラメータースイープを実行するのは非常に簡単です。以下の3つのシンプルなステップだけです：

1. **スイープの定義:** まず、探索するパラメーター、使用する探索戦略、最適化するメトリクスなどを指定する辞書のようなオブジェクトを作成します。

2. **スイープの初期化:** 1行のコードでスイープを初期化し、スイープ構成の辞書を渡します：
`すweep_id = wandb.sweep(sweep_config)`

3. **スイープエージェントの実行:** 同じく1行のコードで、`wandb.agent()`を呼び、`sweep_id`とモデルのアーキテクチャーと訓練を定義する関数を渡します：
`wandb.agent(sweep_id, function=train)`

以上です！これでハイパーパラメータースイープの実行が完了です！

次のノートブックで、これらの3ステップを詳細に解説します。

このノートブックをフォークして、パラメーターを微調整したり、自分のデータセットでモデルを試してみることを強くお勧めします！

### リソース
- [Sweeps docs →](https://docs.wandb.com/library/sweeps)
- [コマンドラインからの起動 →](https://www.wandb.com/articles/hyperparameter-tuning-as-easy-as-1-2-3)

```python
!pip install wandb -qU
```

```python
import wandb
wandb.login()
```

## 1. スイープの定義

Weights & Biasesのスイープは、ほんの数行のコードで、思い通りにスイープを設定するための強力な手段を提供します。スイープ設定は[辞書またはYAMLファイル](https://docs.wandb.ai/guides/sweeps/configuration)として定義できます。

一緒にいくつかの設定を見てみましょう：
* **メトリクス** – スイープが最適化しようとするメトリクスです。メトリクスには`name`（このメトリクスはトレーニングスクリプトでログに記録する必要があります）と`goal`（`maximize`または`minimize`）を指定できます。
* **探索戦略** – `"method"`キーを使って指定します。いくつかの異なる探索戦略をサポートしています。
  * **Grid Search** – ハイパーパラメーター値のすべての組み合わせを繰り返します。
  * **Random Search** – ハイパーパラメーター値のランダムな組み合わせを繰り返します。
  * **ベイズ探索** – ハイパーパラメーターをメトリクススコアの確率にマッピングする確率モデルを作成し、メトリクスの改善確率が高いパラメーターを選択します。ベイズ最適化の目的は、ハイパーパラメーターの値選びに多くの時間を費やすことですが、試すハイパーパラメーター数を少なくすることです。
* **パラメーター** – ハイパーパラメータ名、離散値、範囲、または各イテレーションで値を取り出す分布を含む辞書です。

すべての設定オプションのリストは[こちら](https://docs.wandb.com/library/sweeps/configuration)で見ることができます。

```python
sweep_config = {
    "method": "random", # グリッドまたはランダム試してみてください
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

## 2. スイープの初期化

`wandb.sweep`を呼び出してスイープコントローラを開始します。これは、パラメーターの`設定`を問い合わせるすべてのエージェントに提供し、`metrics`のパフォーマンスを`wandb`のログを介して返すことを期待します。

```python
sweep_id = wandb.sweep(sweep_config, project="XGBoost-sweeps")
```

### トレーニングプロセスの定義
スイープを実行する前に、モデルを作成しトレーニングする関数を定義する必要があります。これは、ハイパーパラメーターの値を受け取り、メトリクスを出力する関数です。

また、`wandb`をスクリプトに統合する必要があります。主に3つのコンポーネントがあります：
* `wandb.init()` – 新しいW&Bのrunを初期化します。各runはトレーニングスクリプトの単一の実行です。
* `wandb.config` – すべてのハイパーパラメーターをconfigオブジェクトに保存します。これにより[当社のアプリ](https://wandb.ai)を使用して、ハイパーパラメーター値によるrunのソートと比較が可能になります。
* `wandb.log()` – メトリクスやカスタムオブジェクトをログに記録します。これらは画像、動画、音声ファイル、HTML、プロット、ポイントクラウドなどを含むことができます。

また、データのダウンロードも必要です：

```python
!wget https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
```

```python
# Pima Indiansデータセット用のXGBoostモデル
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# データのロード
def train():
  config_defaults = {
    "booster": "gbtree",
    "max_depth": 3,
    "learning_rate": 0.1,
    "subsample": 1,
    "seed": 117,
    "test_size": 0.33,
  }

  wandb.init(config=config_defaults)  # スイープ中にデフォルト値が上書きされます
  config = wandb.config

  # データをロードし、予測子とターゲットに分割
  dataset = loadtxt("pima-indians-diabetes.data.csv", delimiter=",")
  X, Y = dataset[:, :8], dataset[:, 8]

  # データをトレーニングセットとテストセットに分割
  X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                      test_size=config.test_size, 
                                                      random_state=config.seed)

  # トレーニングセット上でモデルをフィット
  model = XGBClassifier(booster=config.booster, max_depth=config.max_depth,
                        learning_rate=config.learning_rate, subsample=config.subsample)
  model.fit(X_train, y_train)

  # テストセット上で予測
  y_pred = model.predict(X_test)
  predictions = [round(value) for value in y_pred]

  # 予測を評価
  accuracy = accuracy_score(y_test, predictions)
  print(f"Accuracy: {accuracy:.0%}")
  wandb.log({"accuracy": accuracy})
```

## 3. エージェントによるスイープの実行

ここで`wandb.agent`を呼び出してスイープを開始します。

`wandb.agent`をW&Bにログイン済みの任意のマシンで呼び出すことができ、
- `sweep_id`
- データセットおよび`train`関数

があれば、スイープに参加します！

> _注意_: `random`スイープはデフォルトで無限に実行されます。
新しいパラメータの組み合わせを次々試し続けますが、
[アプリUIからスイープを停止する](https://docs.wandb.ai/ref/app/features/sweeps)まで続きます。
完了するrunの総数を`agent`に提供することでこれを防ぐことができます。

```python
wandb.agent(sweep_id, train, count=25)
```

## 結果の可視化

スイープが完了したら、結果を確認する時間です。

Weights & Biasesは自動的にいくつかの有用なプロットを生成します。

### 並列座標プロット

このプロットはハイパーパラメーターの値をモデルのメトリクスにマッピングします。最も良いモデルパフォーマンスをもたらしたハイパーパラメーターの組み合わせを特定するのに役立ちます。

このプロットは、学習器としてツリーを使用することが
わずかに良い結果をもたらすことを示しているようです。
しかし、線形モデルを使用することが圧倒的に劣るわけではありません。

![sweeps_xgboost](/images/tutorials/xgboost_sweeps/sweeps_xgboost2.png)

### ハイパーパラメーターの重要性プロット

ハイパーパラメーターの重要性プロットは、メトリクスに最も大きな影響を与えたハイパーパラメーターの値を示します。

相関（線形予測変数として処理）と特徴の重要性（結果に基づいてランダムフォレストをトレーニングした後）を報告し、
どのパラメーターが最も大きな影響を与え、その影響が正であるか負であるかを確認できます。

このチャートを読むと、上記の並列座標チャートで見られた傾向の定量的確認が得られます：
検証精度に対する最大の影響は学習器の選択から来ており、`gblinear`学習器は一般的に`gbtree`学習器よりも劣っていることがわかります。

![sweeps_xgboost](/images/tutorials/xgboost_sweeps/sweeps_xgboost3.png)

これらの可視化は、高価なハイパーパラメーター最適化の実行時間とリソースを節約し、最も重要なパラメータ（および値の範囲）に注目して、さらに探求する価値があるかどうかを判断するのに役立ちます。