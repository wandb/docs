---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# Launchジョブを作成する
<CTAButtons colabLink="https://colab.research.google.com/drive/1wX0OSVxZJDHRsZaOaOEDx-lLUrO1hHgP"/>

LaunchジョブはW&B runsを再現するためのブループリントです。ジョブは、ワークロードを実行するために必要なソースコード、依存関係、および入力をキャプチャするW&B Artifactsです。

`wandb launch`コマンドを使用してジョブを作成および実行します。

:::info
ジョブを作成して実行せずに保存するには、`wandb job create`コマンドを使用します。詳細は[コマンドリファレンスドキュメント](../../ref/cli/wandb-job/wandb-job-create.md)を参照してください。
:::

## Gitジョブ

W&B Launchを使用して、リモートgitリポジトリの特定のコミット、ブランチ、またはタグからコードや他のトラッキングされたアセットをクローンするGitベースのジョブを作成できます。`--uri`または`-u`フラグを使用してコードを含むURIを指定し、オプションで`--build-context`フラグを使用してサブディレクトリーを指定します。

次のコマンドを使用して、gitリポジトリから「hello world」ジョブを実行します：

```bash
wandb launch --uri "https://github.com/wandb/launch-jobs.git" --build-context jobs/hello_world --dockerfile Dockerfile.wandb --project "hello-world" --job-name "hello-world" --entry-point "python job.py"
```

このコマンドは以下を行います：
1. [W&B Launch jobsリポジトリ](https://github.com/wandb/launch-jobs)を一時ディレクトリーにクローンします。
2. **hello**プロジェクトに**hello-world-git**という名前のジョブを作成します。このジョブはリポジトリのデフォルトブランチの最新コミットに関連付けられます。
3. `jobs/hello_world`ディレクトリーと`Dockerfile.wandb`からコンテナイメージをビルドします。
4. コンテナを起動し、`python job.py`を実行します。

特定のブランチまたはコミットハッシュからジョブをビルドするには、`-g`、`--git-hash`引数を追加します。引数の完全なリストについては、`wandb launch --help`を実行してください。

### リモートURL形式

Launchジョブに関連付けられたgitリモートは、HTTPSまたはSSH URLのいずれかです。URLの種類は、ジョブのソースコードを取得するために使用されるプロトコルを決定します。

| リモートURLタイプ | URL形式 | アクセスと認証の要件 |
| ----------| ------------------- | ------------------------------------------ |
| https      | `https://github.com/organization/repository.git`  | gitリモートに認証するためのユーザー名とパスワード |
| ssh        | `git@github.com:organization/repository.git` | gitリモートに認証するためのsshキー |

ホスティングプロバイダーによって正確なURL形式は異なることに注意してください。`wandb launch --uri`で作成されたジョブは、提供された`--uri`で指定された転送プロトコルを使用します。

## コードアーティファクトジョブ

ジョブは、W&B Artifactに保存された任意のソースコードから作成できます。ローカルディレクトリーを`--uri`または`-u`引数で使用して、新しいコードアーティファクトとジョブを作成します。

まず、空のディレクトリーを作成し、次の内容のPythonスクリプト`main.py`を追加します：

```python
import wandb

with wandb.init() as run:
    run.log({"metric": 0.5})
```

次の内容のファイル`requirements.txt`を追加します：

```txt
wandb>=0.17.1
```

ディレクトリーをコードアーティファクトとしてログし、次のコマンドでジョブを起動します：

```bash
wandb launch --uri . --job-name hello-world-code --project launch-quickstart --entry-point "python main.py"
```

上記のコマンドは以下を行います：
1. 現在のディレクトリーを`hello-world-code`という名前のコードアーティファクトとしてログします。
2. `launch-quickstart`プロジェクトに`hello-world-code`という名前のジョブを作成します。
3. 現在のディレクトリーとLaunchのデフォルトDockerfileからコンテナイメージをビルドします。デフォルトDockerfileは`requirements.txt`ファイルをインストールし、エントリーポイントを`python main.py`に設定します。

## イメージジョブ

また、事前に作成されたDockerイメージからジョブをビルドすることもできます。これは、すでに確立されたビルドシステムがある場合や、コードやジョブの要件を調整する予定がないが、ハイパーパラメーターや異なるインフラストラクチャースケールで実験したい場合に便利です。

イメージはDockerレジストリからプルされ、指定されたエントリーポイントまたはデフォルトのエントリーポイントで実行されます。Dockerイメージからジョブを作成して実行するには、`--docker-image`オプションに完全なイメージタグを渡します。

事前に作成されたイメージからシンプルなジョブを実行するには、次のコマンドを使用します：

```bash
wandb launch --docker-image "wandb/job_hello_world:main" --project "hello-world"           
```

## 自動ジョブ作成

W&Bは、Launchで作成されていないrunでも、トラッキングされたソースコードがあるrunに対して自動的にジョブを作成してトラッキングします。runがトラッキングされたソースコードを持つと見なされる条件は以下のいずれかです：
- runに関連付けられたgitリモートとコミットハッシュがある
- runがコードアーティファクトをログした（詳細は[`Run.log_code`](../../ref/python/run.md#log_code)を参照）
- runが`WANDB_DOCKER`環境変数をイメージタグに設定してDockerコンテナで実行された

LaunchジョブがW&B runによって自動的に作成された場合、gitリモートURLはローカルgitリポジトリから推測されます。

### Launchジョブ名

デフォルトでは、W&Bは自動的にジョブ名を生成します。ジョブ名は、ジョブの作成方法（GitHub、コードアーティファクト、またはDockerイメージ）に応じて生成されます。あるいは、環境変数やW&B Python SDKを使用してLaunchジョブの名前を定義することもできます。

次の表は、ジョブソースに基づくデフォルトのジョブ命名規則を示しています：

| ソース        | 命名規則                       |
| ------------- | --------------------------------------- |
| GitHub        | `job-<git-remote-url>-<path-to-script>` |
| コードアーティファクト | `job-<code-artifact-name>`              |
| Dockerイメージ  | `job-<image-name>`                      |

W&B環境変数またはW&B Python SDKを使用してジョブに名前を付けます

<Tabs
defaultValue="env_var"
values={[
{label: '環境変数', value: 'env_var'},
{label: 'W&B Python SDK', value: 'python_sdk'},
]}>
<TabItem value="env_var">

`WANDB_JOB_NAME`環境変数を設定して、希望するジョブ名を指定します。例えば：

```bash
WANDB_JOB_NAME=awesome-job-name
```

  </TabItem>
  <TabItem value="python_sdk">

`wandb.Settings`を使用してジョブの名前を定義します。その後、このオブジェクトを`wandb.init`でW&Bを初期化するときに渡します。例えば：

```python
settings = wandb.Settings(job_name="my-job-name")
wandb.init(settings=settings)
```

  </TabItem>
</Tabs>

:::note
Dockerイメージジョブの場合、バージョンエイリアスは自動的にジョブにエイリアスとして追加されます。
:::

## コンテナ化

ジョブはコンテナ内で実行されます。イメージジョブは事前にビルドされたDockerイメージを使用し、Gitおよびコードアーティファクトジョブはコンテナビルドステップを必要とします。

ジョブのコンテナ化は、`wandb launch`への引数およびジョブソースコード内のファイルでカスタマイズできます。

### ビルドコンテキスト

ビルドコンテキストとは、コンテナイメージをビルドするためにDockerデーモンに送信されるファイルとディレクトリーのツリーを指します。デフォルトでは、Launchはジョブソースコードのルートをビルドコンテキストとして使用します。サブディレクトリーをビルドコンテキストとして指定するには、ジョブを作成および起動する際に`wandb launch`の`--build-context`引数を使用します。

:::tip
`--build-context`引数は、複数のプロジェクトを含むモノレポに関連するGitジョブで特に便利です。サブディレクトリーをビルドコンテキストとして指定することで、モノレポ内の特定のプロジェクトのためにコンテナイメージをビルドできます。

公式のW&B Launch jobsリポジトリを使用した[上記の例](#git-jobs)で、`--build-context`引数の使用方法を確認してください。
:::

### Dockerfile

Dockerfileは、Dockerイメージをビルドするための指示を含むテキストファイルです。デフォルトでは、Launchは`requirements.txt`ファイルをインストールするデフォルトDockerfileを使用します。カスタムDockerfileを使用するには、`wandb launch`の`--dockerfile`引数でファイルのパスを指定します。

Dockerfileのパスはビルドコンテキストに対して相対的に指定されます。例えば、ビルドコンテキストが`jobs/hello_world`で、Dockerfileが`jobs/hello_world`ディレクトリーにある場合、`--dockerfile`引数は`Dockerfile.wandb`に設定する必要があります。公式のW&B Launch jobsリポジトリを使用した[上記の例](#git-jobs)で、`--dockerfile`引数の使用方法を確認してください。

### 要件ファイル

カスタムDockerfileが提供されていない場合、LaunchはPython依存関係をインストールするためにビルドコンテキストを検索します。ビルドコンテキストのルートに`requirements.txt`ファイルが見つかった場合、Launchはファイルに記載された依存関係をインストールします。そうでない場合、`pyproject.toml`ファイルが見つかった場合、Launchは`project.dependencies`セクションから依存関係をインストールします。