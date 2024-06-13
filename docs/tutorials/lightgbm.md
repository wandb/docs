


# LightGBM

[**こちらのColabノートブックで試してみてください →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Simple_LightGBM_Integration.ipynb)

Weights & Biasesを使用して、機械学習の実験トラッキング、データセットのバージョン管理、プロジェクトのコラボレーションを行いましょう。

勾配ブースティング決定木は、構造化データにおける予測モデルの構築において最先端の手法です。

Microsoftの勾配ブースティングフレームワークである[LightGBM](https://github.com/microsoft/LightGBM)は、xgboostをしのぎ、catboostと並んでGBDTアルゴリズムの主要な選択肢となっています。LightGBMは、トレーニング速度、メモリ使用量、そして扱えるデータセットのサイズにおいてxgboostを上回ります。LightGBMは、トレーニング中に連続特徴量を離散的なビンにバケット化するヒストグラムベースのアルゴリズムを使用してこれを実現しています。

**[W&B + LightGBMのドキュメントはこちら](https://docs.wandb.ai/guides/integrations/boosting)**


## このノートブックの内容
* Weights & BiasesとLightGBMの簡単な統合
* メトリクスのログ取得用の`wandb_callback()`コールバック
* 特徴量重要度プロットのログ取りとモデル保存を可能にする`log_summary()`関数

私たちは、モデルの内部を見ることを非常に簡単にしたいと考えています。そのため、一行のコードでLightGBMの性能を視覚化するのに役立つコールバックを作成しました。

**注意**: _Step_で始まるセクションは、W&Bを統合するために必要なすべてです。

# インストール、インポート、ログイン

## 日常のインストール


```ipython
!pip install -Uq 'lightgbm>=3.3.1'
```


```python
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
```

## Step 0: W&Bをインストール


```ipython
!pip install -qU wandb
```

## Step 1: W&Bのインポートとログイン


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

# lightgbm用にデータセットを作成
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
```

# トレーニング

### Step 2: W&Bのrunを初期化

`wandb.init()`を使用してW&Bのrunを初期化します。設定の辞書を渡すこともできます。 [公式ドキュメントはこちら →](https://docs.wandb.com/library/init)

ML/DLワークフローにおける設定の重要性は否定できません。W&Bはモデルの再現性を確保するために適切な設定へのアクセスを提供します。

[このColabノートブックで設定について学びましょう →](http://wandb.me/config-colab)


```python
# 辞書として設定を指定
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

> モデルをトレーニングしたら、**プロジェクトページ**に戻ってください。

### Step 3: `wandb_callback`を使用してトレーニング


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

### Step 4: `log_summary`を使用して特徴量重要度をログし、モデルをアップロード
`log_summary`は特徴量重要度を計算してアップロードし、（オプションで）トレーニング済みモデルをW&B Artifactsにアップロードします。このため、後で使用することができます。


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

特定のW&B runのログを終了する際には、`wandb.finish()`を呼び出してwandbプロセスをきちんと整理するのが良い習慣です（ノートブックやcolabs使用時のみ必要）。


```python
wandb.finish()
```

# 結果を視覚化

上記の**プロジェクトページ**リンクをクリックして、結果を自動的に視覚化しましょう。

<img src="https://imgur.com/S6lwSig.png" alt="Viz" />


# Sweep入門

Weights & Biases Sweepsを使用して、ハイパーパラメーターの最適化を自動化し、モデルの可能性の空間を探索しましょう。

## [W&B Sweepを使用してXGBoostのハイパーパラメーター最適化をチェック →](http://wandb.me/xgb-colab)

Weights & Biasesでハイパーパラメーター探索を実行するのはとても簡単です。以下の3つのステップだけです。

1. **スイープを定義する:** 検索するパラメーター、検索戦略、最適化指標などを指定する辞書または[YAMLファイル](https://docs.wandb.com/library/sweeps/configuration)を作成します。

2. **スイープを初期化する:** 
`sweep_id = wandb.sweep(sweep_config)`

3. **スイープエージェントを実行する:** 
`wandb.agent(sweep_id, function=train)`

そして、確かに！ハイパーパラメーター探索を実行するにはこれだけです！

<img src="https://imgur.com/SVtMfa2.png" alt="Sweep Result" />


# 例ギャラリー

W&Bを使用してトラッキングおよび視覚化されたプロジェクトの例を、[ギャラリーで見る →](https://app.wandb.ai/gallery)

# 基本設定
1. **Projects**: 複数のrunsをプロジェクトにログして比較します。`wandb.init(project="project-name")`
2. **Groups**: 複数のプロセスまたは交差検証フォールドの場合、各プロセスをrunとしてログし、一緒にグループ化します。`wandb.init(group='experiment-1')`
3. **Tags**: 現在のベースラインやプロダクションモデルをトラッキングするためにタグを追加します。
4. **Notes**: テーブルにメモを入力して、runs間の変更を追跡します。
5. **Reports**: 同僚と進捗状況を共有し、MLプロジェクトのダッシュボードやスナップショットを作成するための素早いメモを残します。

# 高度な設定
1. [環境変数](https://docs.wandb.com/library/environment-variables): マネージドクラスターでトレーニングを実行するためにAPIキーを環境変数に設定します。
2. [オフラインモード](https://docs.wandb.com/library/technical-faq#can-i-run-wandb-offline): `dryrun`モードを使用してオフラインでトレーニングを行い、後で結果を同期します。
3. [オンプレミス](https://docs.wandb.com/self-hosted): W&Bをプライベートクラウドやエアギャップされたサーバーにインストールします。学術機関から企業チームまで、すべての人に対してローカルインストールを提供しています。
4. [Sweeps](https://docs.wandb.com/sweeps): 軽量なツールを使用してハイパーパラメーター検索を迅速に設定します。