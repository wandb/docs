
# XGBoost Sweeps

[**Try in a Colab Notebook here →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Using_W&B_Sweeps_with_XGBoost.ipynb)

Weights & Biasesを使用して、機械学習実験の追跡やデータセットのバージョン管理、プロジェクトのコラボレーションを行いましょう。

<img src="http://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />

ツリーベースのモデルから最高のパフォーマンスを引き出すためには、[適切なハイパーパラメーターを選択する](https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f)必要があります。`early_stopping_rounds`は何回に設定すべきか？木の`max_depth`はどれくらいにすべきか？

高次元のハイパーパラメータースペースを探索して最もパフォーマンスの高いモデルを見つけるのは非常に大変です。ハイパーパラメーター探索は、モデルの頂点を決めるための組織化された効率的な方法を提供します。これにより、ハイパーパラメーターの組み合わせを自動的に探索し、最適な値を見つけ出すことができます。

このチュートリアルでは、Weights & Biasesを使用してXGBoostモデルで高度なハイパーパラメーター探索を3つの簡単なステップで実行する方法を紹介します。

予告編として、以下のプロットをご覧ください：

![sweeps_xgboost](/images/tutorials/xgboost_sweeps/sweeps_xgboost.png)

## Sweeps: 概要

Weights & Biasesを使用してハイパーパラメーター探索を実行するのは非常に簡単です。基本的な手順は3つだけです：

1. **スイープを定義する:** 辞書のようなオブジェクトを作成して、スイープの定義を行います。これは、探索するパラメーター、使用する探索戦略、最適化するメトリクスを指定します。

2. **スイープを初期化する:** 1行のコードでスイープを初期化し、スイープ構成の辞書を渡します：
`sweep_id = wandb.sweep(sweep_config)`

3. **スイープエージェントを実行する:** これも1行のコードで実行し、`sweep_id`とともにモデルのアーキテクチャとトレーニングを定義する関数を渡します：
`wandb.agent(sweep_id, function=train)`

これで、ハイパーパラメーター探索が完成です！

以下のノートブックでは、これらの3つのステップを詳しく解説します。

ぜひノートブックをフォークし、パラメーターを微調整したり、自分のデータセットでモデルを試してみてください！

### リソース
- [Sweeps ドキュメント →](https://docs.wandb.com/library/sweeps)
- [コマンドラインからの起動 →](https://www.wandb.com/articles/hyperparameter-tuning-as-easy-as-1-2-3)

```python
!pip install wandb -qU
```

```python
import wandb
wandb.login()
```

## 1. スイープの定義

Weights & BiasesのSweepsを使用することで、わずか数行のコードでスイープを正確に設定できます。スイープの構成は、[辞書またはYAMLファイル](https://docs.wandb.ai/guides/sweeps/configuration)として定義できます。

いくつかの主要な構成要素を一緒に見ていきましょう。
* **メトリクス** – スイープが最適化を試みるメトリクスです。メトリクスは`name`（このメトリクスはトレーニングスクリプトでログインされるべきもの）と`goal`（`maximize`または`minimize`）を取ります。
* **探索戦略** – `"method"`キーを使用して指定されます。We support several different search strategies with sweeps.
  * **Grid Search** – すべてのハイパーパラメーター値の組み合わせを繰り返し試します。
  * **Random Search** – ランダムに選ばれたハイパーパラメーター値の組み合わせを繰り返し試します。
  * **Bayesian Search** – ハイパーパラメーターとメトリクススコアの確率をマッピングする確率モデルを作成し、メトリクスの改善の高確率のあるパラメーターを選択します。ベイズ最適化の目的は、選択するハイパーパラメーターの値に多くの時間を費やしながら、試すハイパーパラメーターの値を減らすことです。
* **パラメーター** – 名前、範囲、または各反復でそれらの値を引く分布を持つハイパーパラメーター名を含む辞書です。

すべての設定オプションのリストは[こちら](https://docs.wandb.com/library/sweeps/configuration)で確認できます。

```python
sweep_config = {
    "method": "random", # gridまたはrandomを試してみてください
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

`wandb.sweep`を呼び出すことで、スイープコントローラーを起動します。
これは、`parameters`の設定を問い合わせるすべてのものに提供し、`metrics`のパフォーマンスを`wandb`のログを通じて返すことを期待する集中プロセスです。

```python
sweep_id = wandb.sweep(sweep_config, project="XGBoost-sweeps")
```

### トレーニングプロセスの定義
スイープを実行する前に、モデルを作成しトレーニングする関数を定義する必要があります。
この関数は、ハイパーパラメーター値を受け取り、メトリクスを出力します。

また、`wandb`をスクリプトに統合する必要があります。
主要コンポーネントは以下の3つです：
* `wandb.init()` – 新しいW&Bのrunを初期化します。各runはトレーニングスクリプトの単一実行です。
* `wandb.config` – すべてのハイパーパラメーターを設定オブジェクトに保存します。これにより、[W&Bアプリ](https://wandb.ai)を使用して、ハイパーパラメーターの値によってrunをソートおよび比較できます。
* `wandb.log()` – メトリクスとカスタムオブジェクトをログに記録します。これらは画像、ビデオ、音声ファイル、HTML、プロット、ポイントクラウドなどです。

データのダウンロードも忘れないでください：

```python
!wget https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
```

```python
# Pima IndiansデータセットのXGBoostモデル
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

  wandb.init(config=config_defaults)  # デフォルトはスイープ中にオーバーライドされる
  config = wandb.config

  # データを読み込み、予測因子とターゲットに分割する
  dataset = loadtxt("pima-indians-diabetes.data.csv", delimiter=",")
  X, Y = dataset[:, :8], dataset[:, 8]

  # データをトレーニングセットとテストセットに分割
  X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                      test_size=config.test_size,
                                                      random_state=config.seed)

  # トレーニングセットにモデルを適合させる
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

それでは、`wandb.agent`を呼び出してスイープを開始しましょう。

`wandb.agent`はW&Bにログインしている任意のマシンで呼び出すことができ、
- `sweep_id`
- データセットと`train`関数

を持っているそのマシンがスイープに参加します。

> _注意_: `random`スイープはデフォルトでは無限に実行されます。新しいパラメータの組み合わせを試し続けますが、[アプリUIからスイープをオフにする](https://docs.wandb.ai/ref/app/features/sweeps)まで行います。
これを防ぐために、`agent`が完了するrunの総数を指定することができます。

```python
wandb.agent(sweep_id, train, count=25)
```

## 結果の可視化

スイープが終了したら、結果を確認する時間です。

Weights & Biasesは自動的に多くの有用なプロットを生成します。

### 平行座標プロット

このプロットはハイパーパラメーターの値をモデルのメトリクスにマップします。最も良いモデルパフォーマンスを引き出したハイパーパラメーターの組み合わせを見つけるのに役立ちます。

このプロットは、学習者としてツリーを使用することがわずかに、しかし驚異的なほどではなく、単純な線形モデルを使用するよりも優れていることを示しているようです。

![sweeps_xgboost](/images/tutorials/xgboost_sweeps/sweeps_xgboost2.png)

### ハイパーパラメーターの重要性プロット

ハイパーパラメーターの重要性プロットは、メトリクスに最も大きな影響を与えたハイパーパラメーター値を示します。

相関（線形予測子として扱う）と特徴量重要度（結果に基づいてランダムフォレストをトレーニングした後）を報告しますので、どのパラメーターが最も大きな影響を与え、その影響が正か負かを確認することができます。

このチャートを見ると、平行座標チャートで見られた傾向の定量的な確認がわかります：
検証精度に最大の影響を与えたのは学習者の選択であり、`gblinear`の学習者よりも`gbtree`の学習者の方が一般的に優れていました。

![sweeps_xgboost](/images/tutorials/xgboost_sweeps/sweeps_xgboost3.png)

これらの可視化は、最も重要であり、それゆえさらに探索する価値のあるパラメーター（および値の範囲）を絞り込むことで、高価なハイパーパラメーター最適化を実行する際に時間とリソースを節約するのに役立ちます。