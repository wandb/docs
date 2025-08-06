---
title: XGBoost Sweeps
menu:
  tutorials:
    identifier: xgboost_sweeps
    parent: integration-tutorials
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Using_W&B_Sweeps_with_XGBoost.ipynb" >}}
W&B で機械学習の実験管理、データセットのバージョン管理、プロジェクトの共同作業をはじめましょう。

{{< img src="/images/tutorials/huggingface-why.png" alt="W&B を使うメリット" >}}

ツリーベースのモデルから最大限の性能を引き出すには、[適切なハイパーパラメーターを選ぶ](https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f)ことが重要です。
`early_stopping_rounds` はいくつにすべきか？tree の `max_depth` は何が最適か？

高次元なハイパーパラメータ空間を探索して最適なモデルを見つけるのはすぐに手に負えなくなります。
ハイパーパラメータスイープは、モデル同士の「総当たり戦」を体系的かつ効率的に実施し、勝者を決める方法を提供します。
スイープは、ハイパーパラメータのさまざまな組み合わせを自動で探索し、最適な値を見つけてくれます。

このチュートリアルでは、W&B を使って XGBoost モデルで高度なハイパーパラメータスイープを簡単に実行する 3 ステップをご紹介します。

雰囲気を知りたい方は、以下のプロットをご覧ください。

{{< img src="/images/tutorials/xgboost_sweeps/sweeps_xgboost.png" alt="sweeps_xgboost" >}}

## Sweeps: 概要

W&B のハイパーパラメータスイープはとても簡単に始められます。ステップはたった 3 つです。

1. **スイープの定義:** 辞書のようなオブジェクトを作成し、スイープするパラメータ、探索手法、最適化したいメトリクスを指定します。

2. **スイープの初期化:** 1 行のコードでスイープを初期化し、スイープ設定の辞書を渡します。  
`sweep_id = wandb.sweep(sweep_config)`

3. **スイープエージェントの実行:** こちらも 1 行のコードで、`wandb.agent()` を呼び出し、`sweep_id` とモデルのアーキテクチャやトレーニングを定義した関数を渡します。  
`wandb.agent(sweep_id, function=train)`

これだけでハイパーパラメータスイープが実行できます。

以下のノートブックでは、これら 3 ステップをさらに詳しく解説します。

このノートブックをフォークして、自分のパラメータで試してみたり、ご自身のデータセットでモデルを試したりしてみてください。

### リソース
- [Sweeps ドキュメント →]({{< relref "/guides/models/sweeps/" >}})
- [コマンドラインからの起動方法 →](https://www.wandb.com/articles/hyperparameter-tuning-as-easy-as-1-2-3)



```python
!pip install wandb -qU
```


```python

import wandb
wandb.login()
```

## 1. スイープの定義

W&B Sweeps を使えば、ほんの数行のコードで柔軟にスイープの詳細をコントロールできます。スイープ設定は、[辞書または YAML ファイル]({{< relref "/guides/models/sweeps/define-sweep-configuration" >}})として定義できます。

主な設定項目を一緒に見ていきましょう：
*   **メトリクス**: スイープが最適化を目指すメトリクスです。`name`（トレーニングスクリプト側でログする必要があります）と `goal`（`maximize` または `minimize`）を指定します。
*   **探索手法**: `"method"` キーで指定します。Sweeps ではいくつかの探索アルゴリズムに対応しています。
  *   **グリッド検索**: すべてのパラメータの組み合わせを総当たりで探索します。
  *   **ランダム検索**: 組み合わせをランダムに選んで探索します。
  *   **ベイズ探索**: ハイパーパラメータとメトリクスのスコア確率をマッピングする確率モデルを作り、スコア改善の確率が高いパラメータを選びます。ベイズ最適化の目的は、より少ない試行回数で有効なパラメータの探索に集中することです。
*   **パラメータ**: ハイパーパラメータ名ごとに、離散値や範囲、あるいは分布を指定し、各試行でその値から選択されます。

詳しくは、[スイープ設定項目の一覧]({{< relref "/guides/models/sweeps/define-sweep-configuration" >}})をご覧ください。


```python
sweep_config = {
    "method": "random", # grid か random で試してみてください
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

`wandb.sweep` を呼び出すと、Sweep Controller（スイープコントローラ）が起動します。  
コントローラは `parameters` の設定を参照しながら、問い合わせてきたエージェントにパラメータを返し、`wandb` でログした `metrics` のパフォーマンスをフィードバックとして受け取ります。


```python
sweep_id = wandb.sweep(sweep_config, project="XGBoost-sweeps")
```

### トレーニング処理の定義
スイープを実行する前に、モデルを構築し学習する関数（ハイパーパラメータを受け取り、メトリクスを返す関数）を定義します。

また、スクリプト内に wandb の組み込みも必要です。
主な 3 コンポーネントは以下です：
*   `wandb.init()`: 新しい W&B Run を初期化します。Run とはトレーニングスクリプト 1 回実行の単位です。
*   `run.config`: すべてのハイパーパラメータを config オブジェクトとして保存します。これにより [W&B のアプリ](https://wandb.ai)で run をハイパーパラメータ別にソート・比較できます。
*   `run.log()`: メトリクスや画像、動画、音声、HTML、プロット、ポイントクラウドなどのカスタムオブジェクトをログします。

データのダウンロードも必要です：


```python
!wget https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
```


```python
# Pima Indians データセット用の XGBoost モデル
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# データの読み込み
def train():
  config_defaults = {
    "booster": "gbtree",
    "max_depth": 3,
    "learning_rate": 0.1,
    "subsample": 1,
    "seed": 117,
    "test_size": 0.33,
  }

  with wandb.init(config=config_defaults)  as run: # sweep 実行時にデフォルトが上書きされます
    config = run.config

    # データを読み込み、説明変数・目的変数へ分割
    dataset = loadtxt("pima-indians-diabetes.data.csv", delimiter=",")
    X, Y = dataset[:, :8], dataset[:, 8]

    # データを学習用・テスト用に分割
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=config.test_size,
                                                        random_state=config.seed)

    # train データでモデルを学習
    model = XGBClassifier(booster=config.booster, max_depth=config.max_depth,
                          learning_rate=config.learning_rate, subsample=config.subsample)
    model.fit(X_train, y_train)

    # test データで予測
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    # 予測の評価
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.0%}")
    run.log({"accuracy": accuracy})
```

## 3. スイープをエージェントで実行

それでは `wandb.agent` を使ってスイープを開始しましょう。

`wandb.agent` は、W&B にログイン済みで
- `sweep_id`
- データセット、`train` 関数

が備わっている任意のマシンで呼び出すことができます。そのマシンはスイープへ参加します。

> _注意_: `random` スイープは、デフォルトでは永遠に実行され、パラメータの組み合わせをずっと探し続けます（[アプリ UI からスイープを停止]({{< relref "/guides/models/sweeps/sweeps-ui" >}})しない限り）。  
これを防ぐには、`agent` が完了すべき run の合計数（`count`）を指定してください。


```python
wandb.agent(sweep_id, train, count=25)
```

## 結果の可視化


スイープが完了したら、実行結果を確認しましょう。

W&B は便利な可視化プロットを自動生成してくれます。

### パラレルコーディネートプロット

このプロットは、ハイパーパラメータの値とモデルのメトリクスを対応付けて表示します。  
どのハイパーパラメータの組み合わせが最良のモデル性能につながったかを特定するのに便利です。

このプロットを見ると、ツリーを使った learner は若干成績が優れていますが、劇的な差があるわけではなさそうです。

{{< img src="/images/tutorials/xgboost_sweeps/sweeps_xgboost2.png" alt="sweeps_xgboost" >}}

### ハイパーパラメータの重要度プロット

ハイパーパラメータのインポータンスプロットは、どのハイパーパラメータの値がメトリクスに最も大きな影響を与えたかを示します。

相関（線形予測器として扱った場合）と特徴量重要度（結果に基づいてランダムフォレストを学習した場合）の両方を表示し、
どのパラメータが最も影響していたか、それがポジティブ・ネガティブどちらの作用かが分かります。

このチャートを見ると、上記のパラレルコーディネートチャートで見た傾向が数値的にも裏付けられています。
最も検証精度にインパクトがあったのは learner の選択であり、`gblinear` learner は `gbtree` learner よりも全般的に劣っていました。

{{< img src="/images/tutorials/xgboost_sweeps/sweeps_xgboost3.png" alt="sweeps_xgboost" >}}

これらの可視化を使って、重要なパラメータや値の範囲を素早く特定できるので、高コストなハイパーパラメータ探索のリソースや時間を削減し、さらに掘り下げる価値のあるポイントを見極められます。