---
title: XGBoost スイープ
menu:
  tutorials:
    identifier: ja-tutorials-integration-tutorials-xgboost_sweeps
    parent: integration-tutorials
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Using_W&B_Sweeps_with_XGBoost.ipynb" >}}
W&B で機械学習実験管理、データセットのバージョン管理、プロジェクトのコラボレーションを行いましょう。

{{< img src="/images/tutorials/huggingface-why.png" alt="W&B を使うメリット" >}}

ツリー系モデルの性能を最大限に引き出すには、[適切なハイパーパラメーターの選択](https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f)が欠かせません。
`early_stopping_rounds` の回数は？ ツリーの `max_depth` は何が最適？

膨大なハイパーパラメーター空間を探索して最も良いモデルを見つけるのはすぐに手に負えなくなります。
ハイパーパラメーター探索（スイープ）は、複数モデルで効果的に競わせ、勝者を決めるための整理された効率的な方法です。
スイープはハイパーパラメーターの組み合わせを自動で探索し、最適な値を見つけ出してくれます。

このチュートリアルでは、W&B を使って XGBoost モデルに対する高度なハイパーパラメーター探索（スイープ）を、たった3つのステップで実行する方法をご紹介します。

まずは下のプロット例をご覧ください。

{{< img src="/images/tutorials/xgboost_sweeps/sweeps_xgboost.png" alt="sweeps_xgboost" >}}

## Sweeps：概要

W&B でハイパーパラメーター探索（スイープ）を実行するのはとても簡単です。やることはこの3つだけ：

1. **スイープの定義:** どのパラメータを探索するか、どんな戦略で探索するか、最適化したいメトリクスは何か、などを定義した辞書（もしくはそれに似たオブジェクト）を作成します。

2. **スイープを初期化:** たった1行のコードでスイープを初期化し、定義した設定辞書を渡します。
`sweep_id = wandb.sweep(sweep_config)`

3. **スイープエージェントの実行:** これも1行で OK！`wandb.agent()` を呼び、`sweep_id` とモデルのアーキテクチャー（学習内容）を定義した関数を渡します。
`wandb.agent(sweep_id, function=train)`

これだけでハイパーパラメーター探索（スイープ）が動きます。

このノートブック内で、この3つのステップについて詳細に解説します。

ぜひノートブックをフォークして、パラメータを変えたり、ご自身のデータセットでモデルを試してみてください。

### 参考リンク
- [Sweeps ドキュメント →]({{< relref path="/guides/models/sweeps/" lang="ja" >}})
- [コマンドラインからの起動方法 →](https://www.wandb.com/articles/hyperparameter-tuning-as-easy-as-1-2-3)


```python
!pip install wandb -qU
```


```python

import wandb
wandb.login()
```

## 1. スイープの定義

W&B のスイープを使えば、わずかなコードで柔軟に細かくスイープを設定できます。スイープの設定（config）は
[辞書オブジェクトまたは YAML ファイル]({{< relref path="/guides/models/sweeps/define-sweep-configuration" lang="ja" >}})で定義できます。

主な要素は下記のとおりです。
*   **Metric（メトリクス）**: スイープで最適化を目指すメトリクスです。`name`（トレーニングスクリプト内でログされるメトリクス名）、`goal`（`maximize` または `minimize`）を含みます。
*   **探索手法（Search Strategy）**: `"method"` キーで指定します。W&B ではいくつかの探索手法が使えます。
  *   **グリッド検索（Grid Search）**: 全てのハイパーパラメーター組み合わせを試します。
  *   **ランダム検索（Random Search）**: ハイパーパラメーター組み合わせをランダムに選んで試します。
  *   **ベイズ探索（Bayesian Search）**: ハイパーパラメーターとメトリクスのスコア確率を結びつける確率モデルを作り、有望なパラメータを優先的に試します。ベイズ最適化ではパラメータ選択（試行）の効率化を狙い、試行回数自体は減ります。
*   **parameters（パラメータ）**: ハイパーパラメーター名、さらに離散値・範囲・分布など、各イテレーションでどこから値を取り出すかを指定する辞書です。

詳細は[全スイープ設定オプション一覧]({{< relref path="/guides/models/sweeps/define-sweep-configuration" lang="ja" >}})をご参考ください。


```python
sweep_config = {
    "method": "random", # grid か random も試せます
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

`wandb.sweep` を呼ぶことで Sweep コントローラ（中央集約型プロセス）が起動します。
これは `parameters` の設定値をエージェントからのリクエストごとに提供し、
エージェントは `wandb` へメトリクスの実測値を返します。


```python
sweep_id = wandb.sweep(sweep_config, project="XGBoost-sweeps")
```

### トレーニングプロセスの関数を定義しよう
スイープの実行前に、
ハイパーパラメーターを受け取ってモデルを構築・トレーニングし、
メトリクスを返す関数を用意します。

`wandb` をスクリプトに組み込みましょう。
ポイントは3つです：
*   `wandb.init()`: 新しい W&B Run を初期化。1 Run が1回のトレーニング実行となります。
*   `run.config`: 全ハイパーパラメーターを config オブジェクトに保存します。これで [W&B アプリ](https://wandb.ai) 上で run をハイパーパラメーターごとに比較・ソートできます。
*   `run.log()`: メトリクスや画像・ビデオ・音声・HTML・グラフ・ポイントクラウドなどのカスタムオブジェクトをログします。

また、データのダウンロードも必要です：


```python
!wget https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
```


```python
# Pima Indians データセット用の XGBoost モデル
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

  with wandb.init(config=config_defaults)  as run: # sweep 実行時、デフォルト値は上書きされます
    config = run.config

    # データのロードと特徴量・ターゲットへの分割
    dataset = loadtxt("pima-indians-diabetes.data.csv", delimiter=",")
    X, Y = dataset[:, :8], dataset[:, 8]

    # 学習用/テスト用の分割
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=config.test_size,
                                                        random_state=config.seed)

    # 学習実行
    model = XGBClassifier(booster=config.booster, max_depth=config.max_depth,
                          learning_rate=config.learning_rate, subsample=config.subsample)
    model.fit(X_train, y_train)

    # 予測
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    # 精度を評価
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.0%}")
    run.log({"accuracy": accuracy})
```

## 3. エージェントでスイープを実行

それでは `wandb.agent` でスイープを開始します。

`wandb.agent` は W&B にログインした任意のマシン上で実行できます。
- `sweep_id`
- データセットと `train` 関数

これらがあれば、そのマシンは sweep に参加できます。

> _注意_: `random` スイープの場合、デフォルトでは永遠に実行され続けます（つまりずっと新しいパラメータ組み合わせを探索し続けます）。
この動作を止めたい場合は、[アプリの UI から sweep を手動で停止]({{< relref path="/guides/models/sweeps/sweeps-ui" lang="ja" >}})するか、
またはエージェントが実行すべき run 数（`count`）をあらかじめ指定しましょう。


```python
wandb.agent(sweep_id, train, count=25)
```

## 結果を可視化しよう


スイープが終わったら、次は結果の確認です。

W&B は役立つ各種可視化プロットを自動生成してくれます。

### Parallel coordinates plot（パラレル座標プロット）

このプロットはハイパーパラメーターの値とモデルメトリクスを可視化します。  
最良のモデルを生み出したハイパーパラメーターの組み合わせを発見するのに便利です。

この図からは、学習器として tree（決定木）を使う方が、
シンプルな線形モデルの場合よりも若干良い結果を出していることがわかります。

{{< img src="/images/tutorials/xgboost_sweeps/sweeps_xgboost2.png" alt="sweeps_xgboost" >}}

### ハイパーパラメータのインポータンスプロット

ハイパーパラメータのインポータンスプロットは
どのハイパーパラメータ値がメトリクスへ最も大きな影響を与えたかを示します。

ここでは、相関（線形予測器としての扱い）と
ランダムフォレストによる特徴量重要度（あなたの result を学習した上で）を両方表示することで、
どのパラメータが一番効果的だったのか、またその効果がプラスなのかマイナスなのかも読み取れます。

このグラフからも、先ほどのパラレル座標チャートで見た傾向が定量的に裏付けられています。
最もバリデーション精度へ強く影響したのは
学習器の種類で、`gblinear` より `gbtree` の方が概して良い結果になりました。

{{< img src="/images/tutorials/xgboost_sweeps/sweeps_xgboost3.png" alt="sweeps_xgboost" >}}

これらの可視化によって、最も重要なパラメータや値の範囲を絞り込むことで、
ハイパーパラメーター最適化にかかる時間・資源を節約し、より有意義な追加探索の指針とすることができます。