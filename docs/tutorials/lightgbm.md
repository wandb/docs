
# LightGBM

[**Colabノートブックで試す →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Simple_LightGBM_Integration.ipynb)

Weights & Biases を使用して機械学習の実験管理、データセットのバージョン管理、プロジェクトの共同作業を行いましょう。

勾配ブースティング決定木(GDBT)は、構造化データの予測モデルを構築する際の最新の技術です。

Microsoftの勾配ブースティングフレームワークである [LightGBM](https://github.com/microsoft/LightGBM) は、xgboostを退け、GDBTアルゴリズムのスタンダードとなりました（catboostと共に）。LightGBMはトレーニング速度、メモリ使用量、処理可能なデータセットのサイズにおいてxgboostを上回ります。これは、トレーニング中に連続する特徴量を離散的なビンにバケット化するヒストグラムベースのアルゴリズムを使用することで実現されています。

**[W&B + LightGBM のドキュメントはこちら](https://docs.wandb.ai/guides/integrations/boosting)**

## このノートブックでカバーする内容
* Weights & Biases と LightGBM の簡単なインテグレーション
* メトリクスのログを記録するための `wandb_callback()` コールバック
* 特徴量のインポータンスプロットをログに記録し、モデルの保存をW&Bに対応させる `log_summary()` 関数

モデルの仕組みを簡単に可視化できるように、LightGBMのパフォーマンスを1行のコードで視覚化するコールバックを構築しました。

**注**: _Step_ から始まるセクションは、W&Bを統合するために必要なすべてです。

# インストール、インポート、ログイン

## よく見かけるもの


```ipython
!pip install -Uq 'lightgbm>=3.3.1'
```


```python
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
```

## ステップ0：W&Bのインストール


```ipython
!pip install -qU wandb
```

## ステップ1：W&Bのインポートとログイン


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
# データセットの読み込みまたは作成
df_train = pd.read_csv("regression.train", header=None, sep="\t")
df_test = pd.read_csv("regression.test", header=None, sep="\t")

y_train = df_train[0]
y_test = df_test[0]
X_train = df_train.drop(0, axis=1)
X_test = df_test.drop(0, axis=1)

# lightgbmのデータセットを作成
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
```

# トレーニング

### ステップ2：ゲームを初期化

`wandb.init()` を使用してW&Bのrunを初期化します。また、設定の辞書を渡すこともできます。 [公式ドキュメントはこちら→](https://docs.wandb.com/library/init)

ML/DLワークフローにおける設定の重要性は否定できません。W&Bは、モデルを再現するために必要な設定に迅速にアクセスできるようにします。

[このColabノートブックで設定の詳細を確認する→](http://wandb.me/config-colab)


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

> モデルのトレーニングが完了したら、**プロジェクトページ**をクリックしてください。

### ステップ3：`wandb_callback`でトレーニング


```python
# トレーニング
# lightgbmのコールバックを追加
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

### ステップ4：`log_summary`で特徴量のインポータンスをログに記録し、モデルをアップロード
`log_summary` を使用して特徴量のインポータンスを計算・アップロードし、(オプションで)トレーニングしたモデルをW&B Artifactsにアップロードします。


```python
log_summary(gbm, save_model_checkpoint=True)
```

# 評価


```python
# 予測
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

# 評価
print("予測のRMSEは:", mean_squared_error(y_test, y_pred) ** 0.5)
wandb.log({"rmse_prediction": mean_squared_error(y_test, y_pred) ** 0.5})
```

特定のW&B runのログ記録が完了したら、`wandb.finish()`を呼び出してwandbプロセスを整理するのが良い習慣です（ノートブックやColabを使用する場合のみ必要）。


```python
wandb.finish()
```

# 結果を視覚化

上記の**プロジェクトページ**リンクをクリックして、結果を自動的に視覚化してください。

<img src="https://imgur.com/S6lwSig.png" alt="Viz" />


# Sweep 101

Weights & Biases Sweepsを使用してハイパーパラメータの最適化を自動化し、可能なモデルの空間を探索しましょう。

## [XGBoostを使用したハイパーパラメータ最適化の詳細はこちら $\rightarrow$](http://wandb.me/xgb-colab)

Weights & Biasesを使用してハイパーパラメータsweepを実行するのは非常に簡単です。以下の3つの簡単なステップで実行できます：

1. **sweepの定義:** 検索するパラメータ、検索戦略、最適化指標などを指定する辞書または[YAMLファイル](https://docs.wandb.com/library/sweeps/configuration)を作成します。

2. **sweepの初期化:** 
`sweep_id = wandb.sweep(sweep_config)`

3. **sweepエージェントの実行:** 
`wandb.agent(sweep_id, function=train)`

そして、これですべてです！ハイパーパラメータsweepの実行が完了します。

<img src="https://imgur.com/SVtMfa2.png" alt="Sweep Result" />


# 例ギャラリー

W&Bでトラッキングおよび視覚化されたプロジェクトの例は、[ギャラリー →](https://app.wandb.ai/gallery)でご覧いただけます。

# 基本設定
1. **Projects**: プロジェクトに複数のrunをログして比較。`wandb.init(project="project-name")`
2. **Groups**: 複数のプロセスまたは交差検証のフォルドを記録し、それらをグループ化。`wandb.init(group='experiment-1')`
3. **Tags**: 現在のベースラインモデルやプロダクションモデルをトラッキングするためにタグを追加。
4. **Notes**: テーブルにメモを入力して、run間の変更点をトラッキング。
5. **Reports**: 進捗状況についての簡単なメモを同僚と共有し、MLプロジェクトのダッシュボードやスナップショットを作成。

# 高度な設定
1. [環境変数](https://docs.wandb.com/library/environment-variables): APIキーを環境変数に設定して、管理されたクラスターでトレーニングを実行。
2. [オフラインモード](https://docs.wandb.com/library/technical-faq#can-i-run-wandb-offline): `dryrun` モードを使用してオフライントレーニングを行い、後で結果を同期。
3. [オンプレミス](https://docs.wandb.com/self-hosted): プライベートクラウドまたはエアギャップサーバーにW&Bをインストール。学術機関からエンタープライズチームまで、すべての人向けにローカルインストールを提供。
4. [Sweeps](https://docs.wandb.com/sweeps): ハイパーパラメータ検索を迅速に設定できる軽量ツールでチューニング。
