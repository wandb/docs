---
description: Set W&B environment variables.
displayed_sidebar: ja
---

# 環境変数

<head>
  <title>W&B 環境変数</title>
</head>

自動化された環境でスクリプトを実行している場合、スクリプトが実行される前に設定されている環境変数またはスクリプト内で **wandb** を制御できます。

```bash
# これは秘密であり、バージョン管理には含めないでください
WANDB_API_KEY=$YOUR_API_KEY
# 名前とノートは省略可能
WANDB_NAME="私の最初の実行"
WANDB_NOTES="学習率を小さくし、正則化を増やす。"
```

```bash
# wandb/settingsファイルをチェックインしない場合にのみ必要
WANDB_ENTITY=$username
WANDB_PROJECT=$project
```

```python
# スクリプトをクラウドに同期させたくない場合
os.environ["WANDB_MODE"] = "offline"
```

## 任意の環境変数

これらの任意の環境変数を使って、リモートマシンでの認証の設定などを行ってください。

| 変数名               | 使い方                                  |
| ---------------------- | ---------- |
| **WANDB\_ANONYMOUS**        | "allow"、"never"、または "must" を設定して、シークレットURL付きの匿名のrunを作成できるようにします。                                                    |
| **WANDB\_API\_KEY**         | あなたのアカウントに関連付けられた認証キーを設定します。[アカウント設定ページ](https://app.wandb.ai/settings)でキーを見つけることができます。リモートマシンで `wandb login` が実行されていない場合は、これを設定する必要があります。               |
| **WANDB\_BASE\_URL**        | [wandb/local](../hosting/intro.md) を使用している場合は、この環境変数を `http://YOUR_IP:YOUR_PORT` に設定する必要があります。        |
| **WANDB\_CACHE\_DIR**       | これはデフォルトで\~/.cache/wandbになっていますが、この環境変数で場所を変更できます。                    |
 | **WANDB\_CONFIG\_DIR**      | これはデフォルトで\~/.config/wandbに設定されていますが、この環境変数で場所を上書きできます                             |
| **WANDB\_CONFIG\_PATHS**    | wandb.configにロードするyamlファイルのカンマ区切りリスト。[config](./config.md#file-based-configs)を参照してください。                                          |
| **WANDB\_CONSOLE**          | これを"off"にすることでstdout / stderrのログを無効にします。対応する環境ではデフォルトで"on"です。                                          |
| **WANDB\_DIR**              | 生成されたすべてのファイルを、トレーニングスクリプトに対して相対的な_wandb_ディレクトリーではなく、ここに絶対パスで保存します。_このディレクトリが存在し、プロセスが実行されるユーザーが書き込みできることを確認してください_                  |
| **WANDB\_DISABLE\_GIT**     | wandbがgitリポジトリを探し、最新のコミット/差分を取得するのを防ぎます。      |
| **WANDB\_DISABLE\_CODE**    | これを true に設定すると、wandb がノートブックや git の差分を保存しなくなります。ただし、git リポジトリ内にある場合、現在のコミットは引き続き保存されます。 |
| **WANDB\_DOCKER**           | これを Docker イメージのダイジェストに設定すると、runの復元が可能になります。これは、wandb の docker コマンドで自動的に設定されます。イメージダイジェストは、`wandb docker my/image/name:tag --digest` を実行することで取得できます。 |
| **WANDB\_ENTITY**           | あなたの run に関連するエンティティです。トレーニングスクリプトのディレクトリで `wandb init` を実行した場合、_wandb_ という名前のディレクトリが作成され、デフォルトのエンティティが保存され、ソースコントロールにチェックインできます。そのファイルを作成したくない場合や、ファイルを上書きしたい場合は、環境変数を使用できます。 |
| **WANDB\_ERROR\_REPORTING** | これを false に設定すると、wandb が致命的なエラーをエラートラッキングシステムに記録しなくなります。  |
| **WANDB\_HOST**             | システムで提供されるホスト名を使用したくない場合は、wandbインターフェースに表示されるホスト名に設定してください。                                |
| **WANDB\_IGNORE\_GLOBS**    | カンマで区切った無視するファイルのglobリストに設定してください。これらのファイルはクラウドに同期されません。                              |
| **WANDB\_JOB\_TYPE**        | "training"や"evaluation"のようなジョブタイプを指定して、異なるタイプのrunを示すことができます。詳しくは[grouping](../runs/../runs/grouping.md)を参照してください。               |
| **WANDB\_MODE**             | これを"offline"に設定すると、wandbはrunのメタデータをローカルに保存し、サーバーと同期しません。「disabled」に設定すると、wandbは完全にオフになります。                  |
| **WANDB\_NAME**             | runの人間が読める名前。設定されていない場合は、ランダムに生成されます。                       |
| **WANDB\_NOTEBOOK\_NAME**   | Jupyterで実行している場合、この変数でノートブックの名前を設定できます。自動検出を試みます。|
| **WANDB\_NOTES**            | 実行に関する長いメモ。マークダウンが許可されており、後でUIで編集できます。|
| **WANDB\_PROJECT**          | 実行に関連するプロジェクト。これは `wandb init` でも設定できますが、環境変数が値を上書きします。|
| **WANDB\_RESUME**           | デフォルトでは、_never_ に設定されています。_auto_ に設定すると、wandbは失敗した実行を自動的に再開します。_must_ に設定すると、実行が開始時に存在するように強制されます。常に独自の一意のIDを生成したい場合は、_allow_に設定し、**WANDB\_RUN\_ID**を常に設定してください。 |
| **WANDB\_RUN\_GROUP**       | 実験名を指定して、runを自動的にグループ化します。詳細は[grouping](../runs/grouping.md)を参照してください。                                 |
| **WANDB\_RUN\_ID**          | スクリプトの単一のrunに対応する、プロジェクトごとのグローバルに一意な文字列を設定します。64文字以内である必要があります。すべての非単語文字はダッシュに変換されます。これは、失敗の場合に既存のrunを再開するために使用できます。      |
| **WANDB\_SILENT**           | wandbのログステートメントを無効にするには、これを**true**に設定します。これが設定されると、すべてのログが**WANDB\_DIR**/debug.logに書き込まれます。               |
| **WANDB\_SHOW\_RUN**        | これを**true**に設定すると、オペレーティングシステムが対応している場合、run urlを表示するブラウザが自動的に開きます。        |
| **WANDB\_TAGS**             | runに適用される、カンマで区切られたタグのリストです。                 |
| **WANDB\_USERNAME**         | 実行に関連付けられたチームメンバーのユーザー名。これにより、サービスアカウントAPIキーと共に、自動化された実行をチームメンバーに帰属させることができます。               |
| **WANDB\_USER\_EMAIL**      | 実行に関連付けられたチームメンバーのメールアドレス。これにより、サービスアカウントAPIキーと共に、自動化された実行をチームメンバーに帰属させることができます。            |

## シングラリティ環境

[Singularity](https://singularity.lbl.gov/index.html)でコンテナを実行している場合、上記の変数に**SINGULARITYENV\_**を前置して環境変数を渡すことができます。シングラリティ環境変数の詳細は[こちら](https://singularity.lbl.gov/docs-environment-metadata#environment)で見つけることができます。

## AWSでの実行

AWSでバッチジョブを実行している場合、W&Bの認証情報を使ってマシンを簡単に認証できます。[設定ページ](https://app.wandb.ai/settings)からAPIキーを取得し、[AWSバッチジョブ仕様](https://docs.aws.amazon.com/batch/latest/userguide/job_definition_parameters.html#parameters)のWANDB_API_KEY環境変数を設定してください。

## よくある質問

### 自動化された実行とサービスアカウント

W&Bにログを送信する実行を起動する自動テストや内部ツールがある場合は、チーム設定ページで**Service Account**を作成してください。これにより、自動ジョブ用のサービスAPIキーを使用できるようになります。 サービスアカウントのジョブを特定のユーザーに帰属させたい場合は、**WANDB\_USERNAME**または**WANDB\_USER\_EMAIL** 環境変数を使用できます。

![チーム設定ページで自動ジョブ用のサービスアカウントを作成する](/images/track/common_questions_automate_runs.png)

これは、連続的なインテグレーションや、自動ユニットテストを設定している場合の TravisCI や CircleCI などのツールに便利です。

### 環境変数は `wandb.init()` に渡されたパラメータを上書きしますか？

`wandb.init` に渡された引数は環境よりも優先されます。環境変数が設定されていない場合に、システムデフォルト以外のデフォルトを持ちたい場合は、`wandb.init(dir=os.getenv("WANDB_DIR", my_default_override))` を呼び出すことができます。

### ロギングをオフにする

`wandb offline` コマンドは、環境変数 `WANDB_MODE=offline` を設定します。これにより、あなたのマシンからリモートのwandbサーバーへのデータの同期が停止されます。複数のプロジェクトがある場合、すべてのプロジェクトがW&Bサーバーへのログデータの同期を停止します。

警告メッセージを静かにするには:

```python
import logging

logger = logging.getLogger("wandb")
logger.setLevel(logging.WARNING)
```

### 共有マシン上の複数のwandbユーザー

共有マシンを使用しており、別の人がwandbユーザーの場合、実行が常に適切なアカウントにログされるように簡単に設定できます。WANDB\_API\_KEY環境変数を設定して認証します。環境の中でこれを参照すると、ログイン時に適切な資格情報が得られるか、スクリプトから環境変数を設定できます。

このコマンド `export WANDB_API_KEY=X` を実行して、XをあなたのAPIキーに置き換えます。ログインしている場合、[wandb.ai/authorize](https://app.wandb.ai/authorize) でAPIキーを見つけることができます。