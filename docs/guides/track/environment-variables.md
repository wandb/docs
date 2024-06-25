---
description: W&Bの環境変数を設定する。
displayed_sidebar: default
---


# 環境変数

<head>
  <title>W&B 環境変数</title>
</head>

スクリプトを自動化された環境で実行する際、**wandb** をスクリプトの実行前または実行中に設定された環境変数で制御することができます。

```bash
# これは秘密であり、バージョン管理にチェックインしないでください
WANDB_API_KEY=$YOUR_API_KEY
# 名前とメモは任意
WANDB_NAME="My first run"
WANDB_NOTES="小さい学習率、より多くの正則化。"
```

```bash
# wandb/settings ファイルをチェックインしない場合のみ必要です
WANDB_ENTITY=$username
WANDB_PROJECT=$project
```

```python
# スクリプトをクラウドに同期させたくない場合
os.environ["WANDB_MODE"] = "offline"
```

## 任意の環境変数

これらの任意の環境変数を使用して、リモートマシンでの認証を設定するなどの操作を行えます。

| 変数名                       | 使い方                                  |
| --------------------------- | ---------- |
| **WANDB\_ANONYMOUS**        | ユーザーが秘密のURLで匿名のrunを作成できるように "allow", "never", または "must" を設定します。                                                    |
| **WANDB\_API\_KEY**         | アカウントに関連付けられた認証キーを設定します。キーは[設定ページ](https://app.wandb.ai/settings)で見つけることができます。リモートマシンで `wandb login` が実行されていない場合、これを設定する必要があります。               |
| **WANDB\_BASE\_URL**        | [wandb/local](../hosting/intro.md) を使用している場合、この環境変数を `http://YOUR_IP:YOUR_PORT` に設定します。        |
| **WANDB\_CACHE\_DIR**       | 既定では \~/.cache/wandb に設定されますが、この環境変数でこの場所を上書きできます。                    |
| **WANDB\_CONFIG\_DIR**      | 既定では \~/.config/wandb に設定されますが、この環境変数でこの場所を上書きできます。                             |
| **WANDB\_CONFIG\_PATHS**    | カンマ区切りのyamlファイルのリストをwandb.configにロードします。[config](./config.md#file-based-configs) を参照してください。                                          |
| **WANDB\_CONSOLE**          | stdout / stderr ロギングを無効にするにはこれを "off" に設定します。対応する環境では既定で "on" になります。                                          |
| **WANDB\_DIR**              | 生成されたすべてのファイルを保管する絶対パスを設定します。トレーニングスクリプトに対して相対的な _wandb_ ディレクトリーの代わりにここに保管します。 _このディレクトリーが存在し、プロセスを実行するユーザーが書き込み可能であることを確認してください_                  |
| **WANDB\_DISABLE\_GIT**     | wandbがgitリポジトリーを調査し、最新のコミット/差分をキャプチャするのを防ぎます。      |
| **WANDB\_DISABLE\_CODE**    | wandbがノートブックやgitの差分を保存するのを防ぎたい場合、これをtrueに設定します。 gitリポジトリーにある場合は現在のコミットをまだ保存します。                   |
| **WANDB\_DOCKER**           | runの復元を有効にするため、dockerイメージのダイジェストを設定します。これはwandb dockerコマンドで自動的に設定されます。 `wandb docker my/image/name:tag --digest` を実行してイメージのダイジェストを取得できます。    |
| **WANDB\_ENTITY**           | runに関連付けられたエンティティです。トレーニングスクリプトのディレクトリーで `wandb init` を実行した場合、_wandb_ という名前のディレクトリーが作成され、ソース管理にチェックインできる既定のエンティティが保存されます。そのファイルを作成したくない場合やそのファイルを上書きしたい場合は、環境変数を使用できます。 |
| **WANDB\_ERROR\_REPORTING** | wandbが致命的なエラーをエラートラッキングシステムにログするのを防ぐにはこれをfalseに設定します。                             |
| **WANDB\_HOST**             | システム提供のホスト名を使用したくない場合、wandbインターフェースに表示したいホスト名を設定します。                                |
| **WANDB\_IGNORE\_GLOBS**    | 無視するファイルのglobパターンのカンマ区切りリストを設定します。これらのファイルはクラウドに同期されません。                              |
| **WANDB\_JOB\_NAME**        | `wandb` によって作成されるジョブの名前を指定します。詳細については [ジョブの作成](../launch/create-launch-job.md) をご覧ください。                                                                                                                                                                                                                        |
| **WANDB\_JOB\_TYPE**        | ジョブタイプを指定します。例えば "training" や "evaluation" を指定して異なる種類のrunを示します。詳細は [グループ化](../runs/grouping.md) をご覧ください。               |
| **WANDB\_MODE**             | これを "offline" に設定すると、wandbはrunのメタデータをローカルに保存し、サーバーに同期しません。これを "disabled" に設定すると、wandbは完全にオフになります。                  |
| **WANDB\_NAME**             | runの人間が読める名前です。設定されていない場合、ランダムに生成されます。                       |
| **WANDB\_NOTEBOOK\_NAME**   | jupyterで実行している場合、この変数でノートブックの名前を設定できます。自動検出を試みます。                    |
| **WANDB\_NOTES**            | runに関する長めのメモです。Markdownが許可されており、後でUIで編集できます。                                    |
| **WANDB\_PROJECT**          | runに関連付けられたプロジェクトです。これも `wandb init` で設定できますが、環境変数がその値を上書きします。                               |
| **WANDB\_RESUME**           | 既定では _never_ に設定されています。_auto_ に設定するとwandbは失敗したrunの自動再開を行います。_must_ に設定すると起動時にrunの存在が強制されます。常に自分で一意のIDを生成したい場合は、_allow_ に設定し、常に **WANDB\_RUN\_ID** を設定します。      |
| **WANDB\_RUN\_GROUP**       | runを自動的にグループ化するための実験名を指定します。詳細は [グループ化](../runs/grouping.md) をご覧ください。                                 |
| **WANDB\_RUN\_ID**          | スクリプトの単一のrunに対応する一意の文字列（プロジェクトごと）を設定します。それは64文字以内である必要があります。すべての非単語文字はダッシュに変換されます。失敗の場合に既存のrunを再開するために使用できます。      |
| **WANDB\_SILENT**           | wandbのログステートメントを無音にするにはこれを **true** に設定します。これが設定されている場合、すべてのログは **WANDB\_DIR**/debug.log に書き込まれます。               |
| **WANDB\_SHOW\_RUN**        | オペレーティングシステムがサポートしている場合、runのURLを自動的にブラウザで開くにはこれを **true** に設定します。        |
| **WANDB\_TAGS**             | runに適用されるタグのカンマ区切りリストです。                 |
| **WANDB\_USERNAME**         | runに関連付けられたチームメンバーのユーザー名です。サービスアカウントのAPIキーと共に使用して、チームメンバーの自動化されたrunの付属を有効にします。               |
| **WANDB\_USER\_EMAIL**      | runに関連付けられたチームメンバーのメールアドレスです。サービスアカウントのAPIキーと共に使用して、チームメンバーの自動化されたrunの付属を有効にします。            |

## Singularity 環境

[Singularity](https://singularity.lbl.gov/index.html)でコンテナを実行している場合、上記の環境変数の前に **SINGULARITYENV\_** を付けることで環境変数を渡すことができます。Singularity環境変数の詳細は[こちら](https://singularity.lbl.gov/docs-environment-metadata#environment)をご覧ください。

## AWSで実行する

AWSでバッチジョブを実行する場合、W&Bの資格情報でマシンを認証するのは簡単です。[設定ページ](https://app.wandb.ai/settings)からAPIキーを取得し、WANDB\_API\_KEY 環境変数を [AWSバッチジョブ仕様](https://docs.aws.amazon.com/batch/latest/userguide/job\_definition\_parameters.html#parameters)に設定します。

## よくある質問

### 自動化されたrunとサービスアカウント

自動化されたテストや内部ツールでW&Bにログを記録するrunを起動する場合、チームの設定ページで **Service Account** を作成してください。これにより、自動化されたジョブ用のサービスAPIキーを使用できるようになります。サービスアカウントジョブを特定のユーザーに付属させたい場合、 **WANDB\_USERNAME** または **WANDB\_USER\_EMAIL** 環境変数を使用できます。

![自動化されたジョブ用のサービスアカウントをチームの設定ページで作成する](/images/track/common_questions_automate_runs.png)

これは、TravisCIやCircleCIなどのツールを使用した継続的インテグレーションや自動化された単体テストの設定に役立ちます。

### 環境変数はwandb.init()に渡されたパラメータを上書きしますか？

`wandb.init` に渡される引数は環境に対して優先されます。環境変数が設定されていない場合にシステムの既定値以外の既定値を持ちたい場合は、 `wandb.init(dir=os.getenv("WANDB_DIR", my_default_override))` を呼び出します。

### ロギングをオフにする

`wandb offline` コマンドは、環境変数 `WANDB_MODE=offline` を設定します。この環境変数は、マシンからリモートwandbサーバーへのデータ同期を停止します。複数のプロジェクトがある場合、それらすべてがW&Bサーバーへのログデータの同期を停止します。

警告メッセージを静音化するには:

```python
import logging

logger = logging.getLogger("wandb")
logger.setLevel(logging.WARNING)
```

### 共有マシン上の複数のwandbユーザー

共有マシンを使用しており、他の人がwandbユーザーである場合、runが常に正しいアカウントにログされるようにするのは簡単です。環境変数WANDB\_API\_KEYを設定して認証します。環境にソースされる場合、ログイン時に正しい資格情報を持つようになり、スクリプトから環境変数を設定することもできます。

次のコマンドを実行します：`export WANDB_API_KEY=X` （ここでXはAPIキーです）。ログインしている場合、APIキーは [wandb.ai/authorize](https://app.wandb.ai/authorize) で見つけることができます。