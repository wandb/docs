---
title: XGBoost Sweeps
menu:
  tutorials:
    identifier: ja-tutorials-integration-tutorials-xgboost_sweeps
    parent: integration-tutorials
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Using_W&B_Sweeps_with_XGBoost.ipynb" >}}
W&B を活用して、機械学習の実験管理、データセットのバージョン管理、プロジェクトのコラボレーションを行いましょう。

{{< img src="/images/tutorials/huggingface-why.png" alt="W&B を使う利点" >}}

ツリーベースのモデルの性能を最大限に引き出すには
[適切なハイパーパラメーターを選ぶ](https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f) 必要があります。
`early_stopping_rounds` はいくつにするか？ ツリーの `max_depth` はどのくらいか？

高次元のハイパーパラメーター空間から最も高性能なモデルを探すのは、あっという間に手に負えなくなります。
ハイパーパラメーター探索は、モデル同士の総当たりを整然かつ効率的に行い、勝者を決める方法を提供します。
ハイパーパラメーターの組み合わせを自動で探索して、最適な値を見つけてくれるのです。

このチュートリアルでは、W&B を使って XGBoost モデルに対して高度なハイパーパラメーター探索を 3 ステップで実行する方法を紹介します。

さわりとして、以下のプロットをご覧ください:

{{< img src="/images/tutorials/xgboost_sweeps/sweeps_xgboost.png" alt="sweeps_xgboost" >}}

## Sweeps: 概要

W&B でハイパーパラメーターの sweep を走らせるのはとても簡単です。必要なのは 3 つのステップだけ:

1. すべての **sweep を定義**: sweep を指定する辞書風のオブジェクトを作成します。どのパラメータを探索するか、どの検索戦略を使うか、どのメトリクスを最適化するかを記述します。

2. **sweep を初期化**: 1 行のコードで sweep を初期化し、sweep 設定の辞書を渡します:
`sweep_id = wandb.sweep(sweep_config)`

3. **sweep agent を実行**: こちらも 1 行のコードで、w`andb.agent()` を呼び出し、`sweep_id` と、モデルのアーキテクチャーを定義して学習させる関数を渡します:
`wandb.agent(sweep_id, function=train)`

これでハイパーパラメーター探索の準備は完了です。

以下のノートブックで、この 3 ステップを詳しく見ていきます。

このノートブックをフォークしてパラメータを調整したり、ご自身のデータセットでモデルを試したりすることを強くおすすめします。

### リソース
- [Sweeps ドキュメント →]({{< relref path="/guides/models/sweeps/" lang="ja" >}})
- [コマンドラインからの Launch →](https://www.wandb.com/articles/hyperparameter-tuning-as-easy-as-1-2-3)



```python
!pip install wandb -qU
```


```python

import wandb
wandb.login()
```

## 1. Sweep を定義する

W&B の Sweeps は、数行のコードで思い通りに sweep を設定できる強力な手段を提供します。sweep の設定は
[辞書または YAML ファイル]({{< relref path="/guides/models/sweeps/define-sweep-configuration" lang="ja" >}})として定義できます。

主な要素を一緒に見ていきましょう:
*   **Metric**: sweeps が最適化を目指すメトリクスです。メトリクスには `name`（このメトリクスはトレーニングスクリプトでログされている必要があります）と `goal`（`maximize` または `minimize`）を指定できます。 
*   **Search Strategy**: `"method"` キーで指定します。Sweeps は複数の検索戦略をサポートしています。 
  *   **Grid Search**: ハイパーパラメーター値のあらゆる組み合わせを総当たりで試します。
  *   **Random Search**: ランダムに選んだハイパーパラメーター値の組み合わせを試します。
  *   **Bayesian Search**: ハイパーパラメーターからメトリクススコアの確率への写像を作る確率モデルを構築し、メトリクスの改善確率が高いパラメータを選びます。ベイズ最適化の目的は、試すハイパーパラメーターの数を減らしつつ、より良い値を選ぶことに時間を使うことです。
*   **Parameters**: ハイパーパラメーター名と、その各反復で値を引くための離散値、範囲、または分布を含む辞書です。

詳細は [sweep configuration の全オプション一覧]({{< relref path="/guides/models/sweeps/define-sweep-configuration" lang="ja" >}})をご覧ください。


```python
sweep_config = {
    "method": "random", # grid または random を試す
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

`wandb.sweep` を呼ぶと、Sweep コントローラが起動します —
集中管理されたプロセスで、`parameters` の設定を問い合わせてくるクライアントに提供し、
それらが `wandb` のログ経由で `metrics` の性能を返すことを期待します。


```python
sweep_id = wandb.sweep(sweep_config, project="XGBoost-sweeps")
```

### トレーニングプロセスを定義する
sweep を実行する前に、
モデルを作成して学習させる関数 —
すなわちハイパーパラメーター値を受け取り、メトリクスを出力する関数 —
を定義する必要があります。

また、スクリプトに `wandb` を組み込む必要があります。
主なコンポーネントは 3 つです:
*   `wandb.init()`: 新しい W&B Run を初期化します。各 run はトレーニングスクリプトの 1 回の実行です。
*   `run.config`: すべてのハイパーパラメーターを config オブジェクトに保存します。これにより、[our app](https://wandb.ai) でハイパーパラメーターの値に基づいて run を並べ替えたり比較したりできます。
*   `run.log()`: メトリクスや、画像・動画・音声ファイル・HTML・プロット・点群などのカスタムオブジェクトをログします。

次に、データをダウンロードします:


```python
!wget https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
```


```python
# Pima Indians データセット向けの XGBoost モデル
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

  with wandb.init(config=config_defaults)  as run: # デフォルトは sweep 実行中に上書きされます
    config = run.config

    # データを読み込み、説明変数と目的変数に分割
    dataset = loadtxt("pima-indians-diabetes.data.csv", delimiter=",")
    X, Y = dataset[:, :8], dataset[:, 8]

    # 訓練用とテスト用に分割
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=config.test_size,
                                                        random_state=config.seed)

    # 学習データでモデルを学習
    model = XGBClassifier(booster=config.booster, max_depth=config.max_depth,
                          learning_rate=config.learning_rate, subsample=config.subsample)
    model.fit(X_train, y_train)

    # テストデータで予測
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    # 予測を評価
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.0%}")
    run.log({"accuracy": accuracy})
```

## 3. sweep agent で Sweep を実行する

ここで `wandb.agent` を呼び出して sweep を開始します。

W&B にログインしている任意のマシンで `wandb.agent` を呼べます。そのマシンに
- `sweep_id`
- データセットと `train` 関数
があれば、そのマシンは sweep に参加します。

> _注意_: `random` の sweep はデフォルトで終わりなく実行され、
パラメータの新しい組み合わせを延々と試し続けます —
あるいは [アプリの UI から sweep を停止する]({{< relref path="/guides/models/sweeps/sweeps-ui" lang="ja" >}}) まで。
`agent` に実行させたい run の総数を `count` で指定すれば回避できます。


```python
wandb.agent(sweep_id, train, count=25)
```

## 結果を可視化する


sweep が完了したら、結果を見てみましょう。

W&B は有用なプロットを自動的にいくつも生成します。

### 平行座標プロット

このプロットは、ハイパーパラメーターの値をモデルのメトリクスに対応付けます。どのハイパーパラメーターの組み合わせが最良のモデル性能につながったかを絞り込むのに役立ちます。

このプロットからは、学習器としてツリーを使うと、
劇的とまではいかないものの、
単純な線形モデルを学習器として使うよりもわずかに良好な性能を示すように見えます。

{{< img src="/images/tutorials/xgboost_sweeps/sweeps_xgboost2.png" alt="sweeps_xgboost" >}}

### ハイパーパラメーターのインポータンスプロット

ハイパーパラメーターのインポータンスプロットは、
どのハイパーパラメーターの値がメトリクスに最も影響したかを示します。

相関（線形予測子として扱う）
と特徴量重要度（結果に対してランダムフォレストを学習した後）
の両方を報告するので、
どのパラメータがどれだけ効いたか、
その効果が正だったか負だったかが分かります。

このチャートを読むと、先ほどの平行座標チャートで見た傾向が
定量的に裏付けられていることが分かります:
検証精度に最も影響したのは学習器の選択で、
`gblinear` の学習器は一般に `gbtree` の学習器より劣っていました。

{{< img src="/images/tutorials/xgboost_sweeps/sweeps_xgboost3.png" alt="sweeps_xgboost" >}}

これらの可視化は、重要なパラメータ（とその値の範囲）を素早く見極めることで、コストの高いハイパーパラメーター最適化にかかる時間とリソースの節約に役立ち、さらに深掘りする価値のある部分に集中できるようにしてくれます。