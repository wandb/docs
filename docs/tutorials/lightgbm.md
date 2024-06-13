
# LightGBM

[**Colabノートブックで試す →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Simple_LightGBM_Integration.ipynb)

Weights & Biasesを使用して機械学習の実験トラッキング、データセットバージョン管理、プロジェクトコラボレーションを行いましょう。

勾配ブースティング決定木は、構造化データの予測モデルを構築する際の最先端技術です。

[LightGBM](https://github.com/microsoft/LightGBM)はMicrosoftによる勾配ブースティングフレームワークで、xgboostを凌駕し、catboostと並んでGBDTアルゴリズムの定番となりました。LightGBMはトレーニング速度、メモリ使用量、処理可能なデータセットのサイズでxgboostを上回っています。トレーニング中に連続特徴を離散ビンにバケット化するヒストグラムベースのアルゴリズムを使用することで、これを実現しています。

**[W&B + LightGBMのドキュメントはこちら](https://docs.wandb.ai/guides/integrations/boosting)**


## このノートブックで扱うこと
* Weights & BiasesとLightGBMの簡単な統合。
* メトリクスログのための`wandb_callback()`コールバック
* 特徴重要度プロットをログし、モデルをW&Bに保存するための`log_summary()`関数

私たちは、モデルの内部を簡単に見られるようにするために、LightGBMのパフォーマンスを一行のコードで視覚化するコールバックを作成しました。

**注意**: _Step_で始まるセクションが、W&Bを統合するために必要な全てです。

# インストール、インポート、ログイン

## おなじみの疑わしき者たち


```ipython
!pip install -Uq 'lightgbm>=3.3.1'
```


```python
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
```

## ステップ 0: W&Bをインストール


```ipython
!pip install -qU wandb
```

## ステップ 1: W&Bをインポートし、ログイン


```python
import wandb
from wandb.integration.lightgbm import wandb_callback, log_summary

wandb.login()
```

# データセットのダウンロードと準備



```ipython
!wget https://raw.githubusercontent.com/microsoft/LightGBM/master/examples/regression/regression.train -qq
!wget https://raw.githubusercontent.com/microsoft/LightGBM/master/examples/regression/regression.test -qq
```


```python
# データセットをロードまたは作成
df_train = pd.read_csv("regression.train", header=None, sep="\t")
df_test = pd.read_csv("regression.test", header=None, sep="\t")

y_train = df_train[0]
y_test = df_test[0]
X_train = df_train.drop(0, axis=1)
X_test = df_test.drop(0, axis=1)

# lightgbm用のデータセットを作成
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
```

# トレーニング

### ステップ 2: W&Bのrunを初期化

`wandb.init()`を使ってW&Bのrunを初期化します。設定の辞書を渡すこともできます。[公式ドキュメントはこちら →](https://docs.wandb.com/library/init)

ML/DLワークフローにおいて設定の重要性を否定することはできません。W&Bは、モデルを再現するために必要な正しい設定へのアクセスを保証します。

[このColabノートブックで設定について詳しく見る →](http://wandb.me/config-colab)


```python
# 設定を辞書として指定
params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": ["rmse", "l2", "l1", "huber"],
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": 0,
}

wandb.init(project="my-lightgbm-project", config=params)
```

> モデルをトレーニングした後、**プロジェクトページ**をクリックしてください。

### ステップ 3: `wandb_callback`でトレーニング


```python
# トレーニング
# lightgbmコールバックを追加
gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round=30,
    valid_sets=lgb_eval,
    valid_names=("validation"),
    callbacks=[wandb_callback()],
    early_stopping_rounds=5,
)
```

### ステップ 4: `log_summary`で特徴重要度をログし、モデルをアップロード
`log_summary`は、特徴重要度を計算しアップロードし、（オプションで）訓練されたモデルをW&BのArtifactsにアップロードします。これにより後で使用することができます。


```python
log_summary(gbm, save_model_checkpoint=True)
```

# 評価


```python
# 予測
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

# 評価
print("予測のrmseは:", mean_squared_error(y_test, y_pred) ** 0.5)
wandb.log({"rmse_prediction": mean_squared_error(y_test, y_pred) ** 0.5})
```

特定のW&B runのログを終了したら、`wandb.finish()`を呼び出してwandbプロセスを整理することをお勧めします（ノートブックやColabを使用する場合のみ必要です）


```python
wandb.finish()
```

# 結果の視覚化

上記の**プロジェクトページ**リンクをクリックして、結果が自動的に視覚化されるのを確認してください。

<img src="https://imgur.com/S6lwSig.png" alt="Viz" />


# Sweep 101

Weights & BiasesのSweepsを使用してハイパーパラメーター最適化を自動化し、可能なモデルの空間を探索します。

## [W&B Sweepを使用したXGBoostでのハイパーパラメーター最適化を見る →](http://wandb.me/xgb-colab)

Weights & Biasesを使用したハイパーパラメータースイープを実行するのはとても簡単です。たった3つの簡単なステップがあります：

1. **スイープの定義:** 検索するパラメーター、検索戦略、最適化メトリクスなどを指定する辞書や[YAMLファイル](https://docs.wandb.com/library/sweeps/configuration)を作成します。

2. **スイープの初期化:** 
`sweep_id = wandb.sweep(sweep_config)`

3. **スイープエージェントの実行:** 
`wandb.agent(sweep_id, function=train)`

そして、これでハイパーパラメータースイープの実行は完了です！

<img src="https://imgur.com/SVtMfa2.png" alt="Sweep Result" />


# Example Gallery

W&Bでトラッキングおよび視覚化されたプロジェクトの例を[ギャラリー →](https://app.wandb.ai/gallery)でご覧ください。

# Basic Setup
1. **Projects**: 複数のrunをプロジェクトにログして比較します。`wandb.init(project="project-name")`
2. **Groups**: 複数のプロセスや交差検証フォールドのために、それぞれのプロセスをrunとしてログし、グループ化します。`wandb.init(group='experiment-1')`
3. **Tags**: 現在のベースラインまたはプロダクションモデルを追跡するためにタグを追加します。
4. **Notes**: テーブルにメモを入力して、run間の変更を追跡します。
5. **Reports**: 同僚と共有するための進捗メモを取り、MLプロジェクトのダッシュボードとスナップショットを作成します。

# Advanced Setup
1. [環境変数](https://docs.wandb.com/library/environment-variables): 環境変数にAPIキーを設定して、管理されたクラスターでトレーニングを実行します。
2. [オフラインモード](https://docs.wandb.com/library/technical-faq#can-i-run-wandb-offline): `dryrun`モードを使用してオフライントレーニングを行い、後で結果を同期します。
3. [オンプレ](https://docs.wandb.com/self-hosted): W&Bをプライベートクラウドやエアギャップサーバーにインストールします。学術機関から企業のチームまで、あらゆる人向けのローカルインストールがあります。
4. [Sweeps](https://docs.wandb.com/sweeps): 軽量ツールを使ってハイパーパラメター検索を迅速にセットアップします。