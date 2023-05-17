---
description: Set W&B environment variables.
displayed_sidebar: default
---

# 環境変数

<head>
  <title>W&B 環境変数</title>
</head>

自動化された環境でスクリプトを実行している場合、スクリプトが実行される前に設定された環境変数やスクリプト内で**wandb**を制御できます。

```bash
# これは秘密であり、バージョン管理に登録しないでください
WANDB_API_KEY=$YOUR_API_KEY
# 名前とメモは任意です
WANDB_NAME="My first run"
WANDB_NOTES="学習率を小さくし、正則化を強化"
```

```bash
# wandb/settingsファイルをチェックインしていない場合のみ必要です
WANDB_ENTITY=$username
WANDB_PROJECT=$project
```

```python
# スクリプトをクラウドと同期させたくない場合
os.environ['WANDB_MODE'] = 'offline'
```
## Singularity環境

[Singularity](https://singularity.lbl.gov/index.html)でコンテナを実行している場合は、上記の変数に **SINGULARITYENV\_** を追加することで環境変数を渡すことができます。 Singularity環境変数の詳細は[こちら](https://singularity.lbl.gov/docs-environment-metadata#environment)で確認できます。

## AWSでの実行

AWSでバッチジョブを実行している場合、W&Bの認証情報を使用してマシンを簡単に認証できます。[設定ページ](https://app.wandb.ai/settings)からAPIキーを取得し、[AWSバッチジョブ仕様](https://docs.aws.amazon.com/batch/latest/userguide/jobdefinitionparameters.html#parameters)でWANDB\_API\_KEY環境変数を設定してください。

## よくある質問

### 自動化されたrunとサービスアカウント

W&Bにログを送信する自動化されたテストや内部ツールがある場合は、チーム設定ページで**サービスアカウント**を作成してください。これにより、自動化されたジョブ用のサービスAPIキーを使用できます。サービスアカウントのジョブを特定のユーザーに帰属させたい場合は、**WANDB\_USERNAME**または**WANDB\_USER\_EMAIL**環境変数を使用できます。

![自動化されたジョブ用にチーム設定ページでサービスアカウントを作成する](/images/track/common_questions_automate_runs.png)

これは、自動化されたユニットテストのセットアップにTravisCIやCircleCIなどのツールを使用している場合に役立ちます。

### 環境変数はwandb.init()に渡されるパラメータを上書きしますか？

`wandb.init`に渡される引数は環境よりも優先されます。環境変数が設定されていない場合にシステムデフォルト以外のデフォルトを使用したい場合は、`wandb.init(dir=os.getenv("WANDB_DIR", my_default_override))`を呼び出すことができます。

### ロギングをオフにする

`wandb offline`コマンドは、`WANDB_MODE=offline`という環境変数を設定します。これにより、マシンからリモートwandbサーバーへのデータの同期が停止します。複数のプロジェクトがある場合、すべてのプロジェクトのデータがW&Bサーバーに同期されなくなります。

警告メッセージを消すには：

```python
import logging
logger = logging.getLogger("wandb")
logger.setLevel(logging.WARNING)
```
### 共有マシン上の複数のwandbユーザー

共有マシンを使用しており、別の人がwandbユーザーである場合、runsが常に適切なアカウントにログされるようにするのは簡単です。WANDB_API_KEY環境変数を設定して認証します。環境変数をenvにsourceすると、ログイン時に正しい認証情報が得られますし、スクリプトから環境変数を設定することもできます。

このコマンド `export WANDB_API_KEY=X` を実行します。ここで、XはあなたのAPIキーです。ログインしている場合、APIキーは [wandb.ai/authorize](https://app.wandb.ai/authorize) で見つけることができます。