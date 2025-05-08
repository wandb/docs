---
title: ローンンチジョブを作成する
menu:
  launch:
    identifier: ja-launch-create-and-deploy-jobs-create-launch-job
    parent: create-and-deploy-jobs
url: /ja/guides/launch/create-launch-job
---

{{< cta-button colabLink="https://colab.research.google.com/drive/1wX0OSVxZJDHRsZaOaOEDx-lLUrO1hHgP" >}}

Launch ジョブは、W&B Runs を再現するための設計図です。ジョブは、ワークロードを実行するために必要なソースコード、依存関係、および入力をキャプチャする W&B Artifacts です。

`wandb launch` コマンドでジョブを作成して実行します。

{{% alert %}}
実行に送信せずにジョブを作成するには、`wandb job create` コマンドを使用します。詳細については、[コマンドリファレンスドキュメント]({{< relref path="/ref/cli/wandb-job/wandb-job-create.md" lang="ja" >}}) を参照してください。
{{% /alert %}}

## Git ジョブ

W&B Launch を使って、ソースコードや他の追跡されたアセットをリモート git リポジトリの特定のコミット、ブランチ、またはタグからクローンする Git ベースのジョブを作成できます。`--uri` または `-u` フラグを使用して、コードを含む URI を指定し、オプションとして `--build-context` フラグを使用してサブディレクトリーを指定します。

次のコマンドを使用して git リポジトリから "hello world" ジョブを実行します:

```bash
wandb launch --uri "https://github.com/wandb/launch-jobs.git" --build-context jobs/hello_world --dockerfile Dockerfile.wandb --project "hello-world" --job-name "hello-world" --entry-point "python job.py"
```

このコマンドは次のことを行います:
1. [W&B Launch ジョブリポジトリ](https://github.com/wandb/launch-jobs) を一時ディレクトリーにクローンします。
2. **hello** プロジェクト内に **hello-world-git** という名前のジョブを作成します。このジョブはリポジトリのデフォルトブランチのトップにあるコミットに関連付けられています。
3. `jobs/hello_world` ディレクトリーと `Dockerfile.wandb` からコンテナイメージをビルドします。
4. コンテナを開始し、`python job.py` を実行します。

特定のブランチまたはコミットハッシュからジョブをビルドするには、`-g`、`--git-hash` 引数を追加します。引数の完全なリストについては、`wandb launch --help` を実行してください。

### リモート URL 形式

Launch ジョブに関連付けられた git リモートは、HTTPS または SSH URL のいずれかを使用できます。URL タイプは、ジョブのソースコードを取得するために使用されるプロトコルを決定します。

| リモート URL タイプ | URL 形式 | アクセスと認証の要件 |
| ----------------- | --------- | ------------------------------------------- |
| https             | `https://github.com/organization/repository.git` | git リモートでの認証用のユーザー名とパスワード |
| ssh               | `git@github.com:organization/repository.git`    | git リモートでの認証用の SSH キー            |

正確な URL 形式はホスティングプロバイダーによって異なることに注意してください。`wandb launch --uri` で作成されたジョブは、指定された `--uri` で指定された転送プロトコルを使用します。

## コードアーティファクトジョブ

ジョブは、任意のソースコードから W&B Artifact に保存して作成できます。ローカルディレクトリーを `--uri` または `-u` 引数で使用して、新しいコードアーティファクトとジョブを作成します。

まず、空のディレクトリーを作成し、次の内容を持つ Python スクリプト `main.py` を追加します:

```python
import wandb

with wandb.init() as run:
    run.log({"metric": 0.5})
```

次の内容を持つファイル `requirements.txt` を追加します:

```txt
wandb>=0.17.1
```

ディレクトリーをコードアーティファクトとして記録し、次のコマンドでジョブを起動します:

```bash
wandb launch --uri . --job-name hello-world-code --project launch-quickstart --entry-point "python main.py"
```

前のコマンドは次のことを行います:
1. 現在のディレクトリーを `hello-world-code` という名前のコードアーティファクトとして記録します。
2. `launch-quickstart` プロジェクト内に `hello-world-code` という名前のジョブを作成します。
3. 現在のディレクトリーと Launch のデフォルトの Dockerfile からコンテナイメージをビルドします。デフォルトの Dockerfile は `requirements.txt` ファイルをインストールし、エントリーポイントを `python main.py` に設定します。

## イメージジョブ

別の方法として、事前に作成された Docker イメージからジョブを構築することもできます。これは、すでに ML コード用の確立されたビルドシステムを持っている場合や、ジョブのコードや要件を調整することはほとんどなく、ハイパーパラメーターや異なるインフラストラクチャー規模で実験したい場合に役立ちます。

イメージは Docker レジストリから取得され、指定されたエントリーポイントまたは指定されていない場合はデフォルトのエントリーポイントで実行されます。Docker イメージからジョブを作成して実行するには、`--docker-image` オプションに完全なイメージタグを渡します。

事前に作成されたイメージからシンプルなジョブを実行するには、次のコマンドを使用します:

```bash
wandb launch --docker-image "wandb/job_hello_world:main" --project "hello-world"           
```

## 自動ジョブ作成

W&B は、追跡されたソースコードを持つ任意の Run に対してジョブを自動的に作成して追跡します。この Run が Launch で作成されていなくてもです。Run が追跡されたソースコードを持つと見なされる条件は次の3つのうちのいずれかを満たした場合です:
- Run に関連付けられた git リモートとコミットハッシュがある
- Run がコードアーティファクトを記録した (詳細については [`Run.log_code`]({{< relref path="/ref/python/run.md#log_code" lang="ja" >}}) を参照)
- Run が `WANDB_DOCKER` 環境変数をイメージタグに設定した Docker コンテナで実行された

Git リモート URL は、W&B Run によって自動的に作成された Launch ジョブの場合、ローカルの git リポジトリから推測されます。

### Launch ジョブ名

デフォルトでは、W&B はジョブ名を自動的に生成します。名前は、ジョブの作成方法 (GitHub、コードアーティファクト、Docker イメージ) によって生成されます。別の方法として、環境変数または W&B Python SDK で Launch ジョブの名前を定義できます。

次の表は、ジョブソースに基づくデフォルトのジョブの命名規則について説明しています:

| ソース       | 命名規則                                      |
| ------------- | ----------------------------------------------- |
| GitHub        | `job-<git-remote-url>-<path-to-script>`         |
| Code artifact | `job-<code-artifact-name>`                      |
| Docker image  | `job-<image-name>`                              |

W&B 環境変数または W&B Python SDK でジョブに名前を付けます

{{< tabpane text=true >}}
{{% tab "Environment variable" %}}
`WANDB_JOB_NAME` 環境変数を希望のジョブ名に設定します。例:

```bash
WANDB_JOB_NAME=awesome-job-name
```
{{% /tab %}}
{{% tab "W&B Python SDK" %}}
`wandb.Settings` でジョブの名前を定義します。そして、このオブジェクトを使用して W&B を `wandb.init` で初期化する際に渡します。例:

```python
settings = wandb.Settings(job_name="my-job-name")
wandb.init(settings=settings)
```
{{% /tab %}}
{{< /tabpane >}}

{{% alert %}}
Docker イメージジョブの場合、バージョンエイリアスは自動的にジョブのエイリアスとして追加されます。
{{% /alert %}}

## コンテナ化

ジョブはコンテナ内で実行されます。イメージジョブは事前構築された Docker イメージを使用し、Git およびコードアーティファクトジョブはコンテナビルドステップが必要です。

ジョブのコンテナ化は、`wandb launch` の引数やジョブソースコード内のファイルでカスタマイズできます。

### ビルドコンテキスト

ビルドコンテキストとは、コンテナイメージをビルドするために Docker デーモンに送信されるファイルとディレクトリーのツリーを指します。デフォルトでは、Launch はジョブソースコードのルートをビルドコンテキストとして使用します。サブディレクトリーをビルドコンテキストとして指定するには、ジョブを作成して起動する際に `wandb launch` の `--build-context` 引数を使用します。

{{% alert %}}
`--build-context` 引数は、複数のプロジェクトを含むモノレポを参照する Git ジョブで作業する際に特に便利です。サブディレクトリーをビルドコンテキストとして指定することで、モノレポ内の特定のプロジェクト用のコンテナイメージをビルドできます。

この引数を公式の W&B Launch ジョブリポジトリで使用する方法については、[上記の例]({{< relref path="#git-jobs" lang="ja" >}})をご覧ください。
{{% /alert %}}

### Dockerfile

Dockerfile は、Docker イメージをビルドするための命令を含むテキストファイルです。デフォルトでは、Launch は `requirements.txt` ファイルをインストールするデフォルトの Dockerfile を使用します。カスタム Dockerfile を使用するには、`wandb launch` の `--dockerfile` 引数でファイルのパスを指定します。

Dockerfile のパスはビルドコンテキストに相対的に指定されます。たとえば、ビルドコンテキストが `jobs/hello_world` で、Dockerfile が `jobs/hello_world` ディレクトリーにある場合、`--dockerfile` 引数は `Dockerfile.wandb` に設定されるべきです。この引数を公式の W&B Launch ジョブリポジトリで使用する方法については、[上記の例]({{< relref path="#git-jobs" lang="ja" >}})をご覧ください。

### 要件ファイル

カスタム Dockerfile が提供されていない場合、Launch は Python の依存関係をインストールするためにビルドコンテキストを調べます。ビルドコンテキストのルートに `requirements.txt` ファイルが見つかった場合、Launch はファイルにリストされた依存関係をインストールします。それ以外の場合、`pyproject.toml` ファイルが見つかれば、`project.dependencies` セクションから依存関係をインストールします。