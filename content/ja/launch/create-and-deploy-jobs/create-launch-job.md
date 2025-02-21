---
title: Create a launch job
menu:
  launch:
    identifier: ja-launch-create-and-deploy-jobs-create-launch-job
    parent: create-and-deploy-jobs
url: guides/launch/create-launch-job
---

{{< cta-button colabLink="https://colab.research.google.com/drive/1wX0OSVxZJDHRsZaOaOEDx-lLUrO1hHgP" >}}

Launch ジョブは W&B Runs を再現するための設計図です。ジョブは、ワークロードを実行するために必要なソースコード、依存関係、入力をキャプチャする W&B Artifacts です。

`wandb launch` コマンドを使って、ジョブを作成し実行します。

{{% alert %}}
ジョブを作成するが実行に提出しない場合は、`wandb job create` コマンドを使用します。詳細については、[command reference docs]({{< relref path="/ref/cli/wandb-job/wandb-job-create.md" lang="ja" >}}) を参照してください。
{{% /alert %}}

## Git ジョブ

コードや他のトラッキングされたアセットをリモート git リポジトリーの特定のコミット、ブランチ、タグからクローンして Git ベースのジョブを作成できます。`--uri` または `-u` フラグを使用して、コードを含む URI を指定し、オプションとして `--build-context` フラグでサブディレクトリーを指定します。

以下のコマンドで git リポジトリーから "hello world" ジョブを実行します：

```bash
wandb launch --uri "https://github.com/wandb/launch-jobs.git" --build-context jobs/hello_world --dockerfile Dockerfile.wandb --project "hello-world" --job-name "hello-world" --entry-point "python job.py"
```

このコマンドは以下のことを行います：
1. [W&B Launch jobs リポジトリー](https://github.com/wandb/launch-jobs) を一時ディレクトリーにクローンします。
2. **hello** プロジェクトにジョブ名 **hello-world-git** を作成します。このジョブはリポジトリーのデフォルトブランチの最初のコミットに関連付けられています。
3. `jobs/hello_world` ディレクトリーと `Dockerfile.wandb` からコンテナーイメージをビルドします。
4. コンテナーを開始し、`python job.py` を実行します。

特定のブランチまたはコミットハッシュからジョブをビルドするには、`-g`、`--git-hash` 引数を追加します。全ての引数の一覧は、`wandb launch --help` を実行して確認してください。

### リモートURL形式

Launch ジョブに関連付けられた git リモートは、HTTPS または SSH URL のどちらかです。URL タイプは、ジョブのソースコードを取得するために使用されるプロトコルを決定します。

| リモート URL タイプ   | URL 形式                        | アクセスと認証の要件                                 |
| -------------------- | ------------------------------- | ------------------------------------------ |
| https                | `https://github.com/organization/repository.git`  | username と password による git リモートへの認証 |
| ssh                  | `git@github.com:organization/repository.git` | ssh キーによる git リモートへの認証         |

URL 形式はホスティングプロバイダーによって異なる場合があります。`wandb launch --uri` で作成されたジョブは、指定された `--uri` で指定された転送プロトコルを使用します。

## コードアーティファクトジョブ

ジョブは、W&B Artifactに保存された任意のソースコードから作成できます。ローカルディレクトリーを `--uri` または `-u` 引数と一緒に使用して、新しいコードアーティファクトとジョブを作成します。

まず、空のディレクトリーを作成し、次の内容で `main.py` という名前のPythonスクリプトを追加します：

```python
import wandb

with wandb.init() as run:
    run.log({"metric": 0.5})
```

次に、次の内容で `requirements.txt` ファイルを追加します：

```txt
wandb>=0.17.1
```

次のコマンドでディレクトリーをコードアーティファクトとしてログし、ジョブを起動します：

```bash
wandb launch --uri . --job-name hello-world-code --project launch-quickstart --entry-point "python main.py"
```

前述のコマンドは以下のことを行います：
1. 現在のディレクトリーを `hello-world-code` という名前のコードアーティファクトとしてログします。
2. `launch-quickstart` プロジェクトで `hello-world-code` という名前のジョブを作成します。
3. 現在のディレクトリーとLaunchのデフォルトの Dockerfile からコンテナーイメージをビルドします。デフォルトの Dockerfile は `requirements.txt` ファイルをインストールし、エントリーポイントを `python main.py` に設定します。

## イメージジョブ

また、既製のDockerイメージからジョブをビルドすることもできます。これは、MLコード用の確立されたビルドシステムがすでにある場合や、ジョブのコードや要件を調整する必要がないが、ハイパーパラメーターや異なるインフラストラクチャーのスケールで実験したい場合に便利です。

イメージはDockerレジストリーからプルされ、指定されたエントリーポイントで実行されます。エントリーポイントが指定されていない場合はデフォルトのエントリーポイントが使用されます。 完全なイメージタグを `--docker-image` オプションで渡して、Dockerイメージからジョブを作成し実行します。

次のコマンドで既製のイメージから簡単なジョブを実行します：

```bash
wandb launch --docker-image "wandb/job_hello_world:main" --project "hello-world"           
```

## 自動ジョブ作成

W&B は、Launch で作成されなくても、トラッキングされたソースコードを持つ実行に対して自動的にジョブを作成してトラッキングします。 実行がトラッキングされたソースコードを持っていると見なされる条件は以下の3つのいずれかが満たされる場合です：
- 実行には関連付けられたgitリモートとコミットハッシュがあります
- 実行はコードアーティファクトを記録しました（詳細については [`Run.log_code`]({{< relref path="/ref/python/run.md#log_code" lang="ja" >}}) を参照）
- 実行は `WANDB_DOCKER` 環境変数がイメージタグに設定された Dockerコンテナーで実行されました

ローカルgitリポジトリーから推測されるGitリモートURLは、LaunchジョブがW&B実行で自動的に作成される場合です。

### Launch ジョブ名

デフォルトで W&B は自動的にジョブ名を生成します。ジョブの作成方法（GitHub、コードアーティファクト、またはDockerイメージ）に応じて名前が生成されます。あるいは、W&B Python SDKまたは環境変数を使用してLaunchジョブの名前を定義できます。

次の表は、ジョブソースに基づいてデフォルトで使用されるジョブ命名規則を説明しています：

| ソース          | 命名規則                                    |
| ------------- | -------------------------------------- |
| GitHub       | `job-<git-remote-url>-<path-to-script>` |
| コードアーティファクト | `job-<code-artifact-name>`                |
| Dockerイメージ  | `job-<image-name>`                      |

W&B 環境変数または W&B Python SDK でジョブの名前を付けます：

{{< tabpane text=true >}}
{{% tab "環境変数" %}}
`WANDB_JOB_NAME` 環境変数を好みのジョブ名に設定します。例：

```bash
WANDB_JOB_NAME=awesome-job-name
```
{{% /tab %}}
{{% tab "W&B Python SDK" %}}
`wandb.Settings` でジョブの名前を定義します。次に、このオブジェクトを `wandb.init` を使用して W&B を初期化するときに渡します。例：

```python
settings = wandb.Settings(job_name="my-job-name")
wandb.init(settings=settings)
```
{{% /tab %}}
{{< /tabpane >}}

{{% alert %}}
Docker イメージジョブに対しては、バージョンエイリアスが自動的にジョブへのエイリアスとして追加されます。
{{% /alert %}}

## コンテナ化

ジョブはコンテナー内で実行されます。イメージジョブは既製のDockerイメージを使用しますが、Gitやコードアーティファクトのジョブにはコンテナービルドステップが必要です。

ジョブのコンテナ化は、`wandb launch` の引数やジョブソースコード内のファイルでカスタマイズできます。

### ビルドコンテキスト

ビルドコンテキストとは、コンテナーイメージをビルドするためにDockerデーモンに送信されるファイルとディレクトリーのツリーを指します。デフォルトで、Launch はジョブソースコードのルートをビルドコンテキストとして使用します。ビルドコンテキストとしてサブディレクトリーを指定するには、ジョブを作成して起動する際に `wandb launch` の `--build-context` 引数を使用します。

{{% alert %}}
`--build-context` 引数は、複数のプロジェクトを含むモノレポを参照する Git ジョブで作業するときに特に便利です。サブディレクトリーをビルドコンテキストとして指定することにより、モノレポ内の特定のプロジェクトのためにコンテナーイメージをビルドできます。

上記の [例]({{< relref path="#git-jobs" lang="ja" >}}) を参照して、公式の W&B Launch jobs リポジトリーと一緒に `--build-context` 引数を使用する方法をデモでご確認ください。
{{% /alert %}}

### Dockerfile

Dockerfile は、 Docker イメージをビルドするための指示を含むテキストファイルです。デフォルトで、 Launch は `requirements.txt` ファイルをインストールするデフォルトの Dockerfile を使用します。カスタム Dockerfile を使用するには、`wandb launch` の `--dockerfile` 引数でファイルへのパスを指定します。

Dockerfile パスはビルドコンテキストに対して相対的に指定されます。例えば、ビルドコンテキストが `jobs/hello_world` で、Dockerfile が `jobs/hello_world` ディレクトリーにある場合、`--dockerfile` 引数は `Dockerfile.wandb` に設定されるべきです。公式の W&B Launch jobs リポジトリーと一緒に `--dockerfile` 引数を使用する方法をデモする [例]({{< relref path="#git-jobs" lang="ja" >}}) を参照してください。

### 依存ファイル

カスタム Dockerfile が提供されていない場合、Launch はインストールする Python 依存関係をビルドコンテキストで探します。ビルドコンテキストのルートに `requirements.txt` ファイルが見つかった場合、Launch はファイルにリストされている依存関係をインストールします。そうでない場合、`pyproject.toml` ファイルが見つかれば、`project.dependencies` セクションから依存関係をインストールします。