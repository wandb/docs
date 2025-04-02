---
title: Create a launch job
menu:
  launch:
    identifier: ja-launch-create-and-deploy-jobs-create-launch-job
    parent: create-and-deploy-jobs
url: guides/launch/create-launch-job
---

{{< cta-button colabLink="https://colab.research.google.com/drive/1wX0OSVxZJDHRsZaOaOEDx-lLUrO1hHgP" >}}

Launch jobsは、W&B runを再現するための設計図です。Jobsは、ワークロードを実行するために必要なソース コード、依存関係、および入力をキャプチャするW&B Artifactsです。

`wandb launch` コマンドでjobsを作成および実行します。

{{% alert %}}
実行のために送信せずにjobを作成するには、`wandb job create` コマンドを使用します。詳細については、[コマンドリファレンスドキュメント]({{< relref path="/ref/cli/wandb-job/wandb-job-create.md" lang="ja" >}})を参照してください。
{{% /alert %}}

## Git jobs

W&B Launchを使用して、コードやその他の追跡対象アセットがリモートgitリポジトリの特定のコミット、ブランチ、またはタグからクローンされるGitベースのjobを作成できます。コードを含むURIを指定するには、`--uri`または`-u`フラグを使用し、必要に応じてサブディレクトリーを指定するには、`--build-context`フラグを使用します。

次のコマンドを使用して、gitリポジトリから「hello world」jobを実行します。

```bash
wandb launch --uri "https://github.com/wandb/launch-jobs.git" --build-context jobs/hello_world --dockerfile Dockerfile.wandb --project "hello-world" --job-name "hello-world" --entry-point "python job.py"
```

このコマンドは次のことを行います。
1. [W&B Launch jobs repository](https://github.com/wandb/launch-jobs)を一時ディレクトリーにクローンします。
2. **hello** プロジェクトに**hello-world-git**という名前のjobを作成します。このjobは、リポジトリのデフォルトブランチの先頭にあるコミットに関連付けられています。
3. `jobs/hello_world`ディレクトリーと`Dockerfile.wandb`からコンテナーイメージを構築します。
4. コンテナーを起動し、`python job.py`を実行します。

特定のブランチまたはコミットハッシュからjobを構築するには、`-g`、`--git-hash`引数を追加します。引数の完全なリストについては、`wandb launch --help`を実行してください。

### リモートURLの形式

Launch jobに関連付けられたgitリモートは、HTTPSまたはSSH URLのいずれかになります。URLタイプは、jobソース コードの取得に使用されるプロトコルを決定します。

| リモートURLタイプ| URL形式 | アクセスと認証の要件 |
| ----------| ------------------- | ------------------------------------------ |
| https      | `https://github.com/organization/repository.git`  | gitリモートで認証するためのユーザー名とパスワード |
| ssh        | `git@github.com:organization/repository.git` | gitリモートで認証するためのsshキー |

正確なURL形式は、ホスティングプロバイダーによって異なることに注意してください。`wandb launch --uri`で作成されたjobsは、指定された`--uri`で指定された転送プロトコルを使用します。

## Code artifact jobs

Jobsは、W&B Artifactに保存されている任意のソース コードから作成できます。`--uri`または`-u`引数を持つローカルディレクトリーを使用して、新しいcode artifactとjobを作成します。

まず、空のディレクトリーを作成し、次のコンテンツを含む`main.py`という名前のPythonスクリプトを追加します。

```python
import wandb

with wandb.init() as run:
    run.log({"metric": 0.5})
```

次のコンテンツを含む`requirements.txt`ファイルを追加します。

```txt
wandb>=0.17.1
```

ディレクトリーをcode artifactとして記録し、次のコマンドでjobを起動します。

```bash
wandb launch --uri . --job-name hello-world-code --project launch-quickstart --entry-point "python main.py"
```

上記のコマンドは次のことを行います。
1. 現在のディレクトリーを`hello-world-code`という名前のcode artifactとして記録します。
2. `launch-quickstart`プロジェクトに`hello-world-code`という名前のjobを作成します。
3. 現在のディレクトリーとLaunchのデフォルトのDockerfileからコンテナーイメージを構築します。デフォルトのDockerfileは、`requirements.txt`ファイルをインストールし、エントリポイントを`python main.py`に設定します。

## Image jobs

または、既製のDockerイメージからjobsを構築することもできます。これは、MLコード用の確立された構築システムがすでに存在する場合、またはjobのコードまたは要件を調整する予定はないが、ハイパーパラメーターまたはさまざまなインフラストラクチャースケールを試したい場合に役立ちます。

イメージはDockerレジストリからプルされ、指定されたエントリポイント、またはエントリポイントが指定されていない場合はデフォルトのエントリポイントで実行されます。`--docker-image`オプションに完全なイメージタグを渡して、Dockerイメージからjobを作成および実行します。

既製のイメージから単純なjobを実行するには、次のコマンドを使用します。

```bash
wandb launch --docker-image "wandb/job_hello_world:main" --project "hello-world"           
```

## Automatic job creation

W&Bは、追跡されたソース コードを含むrunに対してjobを自動的に作成および追跡します。これは、Launchでrunが作成されなかった場合でも同様です。Runは、次の3つの条件のいずれかが満たされた場合に、追跡されたソース コードを持っていると見なされます。
- Runに関連付けられたgitリモートとコミットハッシュがある
- Runがcode artifactを記録した（詳細については、[`Run.log_code`]({{< relref path="/ref/python/run.md#log_code" lang="ja" >}})を参照してください）
- Runが、イメージタグに設定された`WANDB_DOCKER`環境変数を持つDockerコンテナーで実行された

Launch jobがW&B runによって自動的に作成された場合、GitリモートURLはローカルgitリポジトリから推測されます。

### Launch job names

デフォルトでは、W&Bはjob名を自動的に生成します。名前は、jobの作成方法（GitHub、code artifact、またはDockerイメージ）に応じて生成されます。または、環境変数またはW&B Python SDKを使用してLaunch jobの名前を定義することもできます。

次の表は、jobソースに基づいてデフォルトで使用されるjob命名規則を示しています。

| ソース        | 命名規則                       |
| ------------- | --------------------------------------- |
| GitHub        | `job-<git-remote-url>-<path-to-script>` |
| Code artifact | `job-<code-artifact-name>`              |
| Docker image  | `job-<image-name>`                      |

W&B環境変数またはW&B Python SDKを使用してjobに名前を付けます。

{{< tabpane text=true >}}
{{% tab "Environment variable" %}}
`WANDB_JOB_NAME`環境変数を優先job名に設定します。次に例を示します。

```bash
WANDB_JOB_NAME=awesome-job-name
```
{{% /tab %}}
{{% tab "W&B Python SDK" %}}
`wandb.Settings`を使用してjobの名前を定義します。次に、`wandb.init`でW&Bを初期化するときに、このオブジェクトを渡します。次に例を示します。

```python
settings = wandb.Settings(job_name="my-job-name")
wandb.init(settings=settings)
```
{{% /tab %}}
{{< /tabpane >}}

{{% alert %}}
docker image jobsの場合、バージョンエイリアスはjobへのエイリアスとして自動的に追加されます。
{{% /alert %}}

## Containerization

Jobsはコンテナー内で実行されます。Image jobsは既製のDockerイメージを使用しますが、Gitおよびcode artifact jobsはコンテナー構築手順を必要とします。

Jobのコンテナー化は、`wandb launch`への引数とjobソース コード内のファイルを使用してカスタマイズできます。

### Build context

構築コンテキストという用語は、コンテナーイメージを構築するためにDockerデーモンに送信されるファイルとディレクトリーのツリーを指します。デフォルトでは、Launchはjobソース コードのルートを構築コンテキストとして使用します。サブディレクトリーを構築コンテキストとして指定するには、jobの作成および起動時に`wandb launch`の`--build-context`引数を使用します。

{{% alert %}}
`--build-context`引数は、複数のプロジェクトを含むモノレポを指すGit jobsを操作する場合に特に役立ちます。サブディレクトリーを構築コンテキストとして指定することで、モノレポ内の特定のプロジェクトのコンテナーイメージを構築できます。

`--build-context`引数を公式のW&B Launch jobsリポジトリで使用する方法のデモについては、[上記の例]({{< relref path="#git-jobs" lang="ja" >}})を参照してください。
{{% /alert %}}

### Dockerfile

Dockerfileは、Dockerイメージを構築するための命令を含むテキストファイルです。デフォルトでは、Launchは`requirements.txt`ファイルをインストールするデフォルトのDockerfileを使用します。カスタムDockerfileを使用するには、`wandb launch`の`--dockerfile`引数を使用してファイルへのパスを指定します。

Dockerfileパスは、構築コンテキストを基準にして指定されます。たとえば、構築コンテキストが`jobs/hello_world`で、Dockerfileが`jobs/hello_world`ディレクトリーにある場合、`--dockerfile`引数は`Dockerfile.wandb`に設定する必要があります。`--dockerfile`引数を公式のW&B Launch jobsリポジトリで使用する方法のデモについては、[上記の例]({{< relref path="#git-jobs" lang="ja" >}})を参照してください。

### Requirements file

カスタムDockerfileが提供されていない場合、LaunchはインストールするPython依存関係の構築コンテキストを検索します。`requirements.txt`ファイルが構築コンテキストのルートに見つかった場合、Launchはそのファイルにリストされている依存関係をインストールします。それ以外の場合、`pyproject.toml`ファイルが見つかった場合、Launchは`project.dependencies`セクションから依存関係をインストールします。
