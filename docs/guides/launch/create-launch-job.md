---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';


# Launch ジョブの作成
<CTAButtons colabLink="https://colab.research.google.com/drive/1wX0OSVxZJDHRsZaOaOEDx-lLUrO1hHgP"/>

LaunchジョブはW&B runsを再現するためのブループリントです。ジョブは、ワークロードを実行するために必要なソースコード、依存関係、および入力をキャプチャするW&B Artifactsです。

`wandb launch`コマンドを使用してジョブを作成および実行します。

:::情報
実行のためにジョブを送信せずにジョブを作成するには、`wandb job create`コマンドを使用します。詳細については、[command reference docs](../../ref/cli/wandb-job/wandb-job-create.md)を参照してください。
:::

## Gitジョブ

W&B Launchを使って、リモートgitリポジトリ内の特定のコミット、ブランチ、またはタグからコードや他のトラックされたアセットをクローンするGitベースのジョブを作成できます。`--uri`または`-u`フラグを使用してコードを含むURIを指定し、オプションで`--build-context`フラグを使用してサブディレクトリを指定します。

次のコマンドを使って、gitリポジトリから"hello world"ジョブを実行します：

```bash
wandb launch --uri "https://github.com/wandb/launch-jobs.git" --build-context jobs/hello_world --dockerfile Dockerfile.wandb --project "hello-world" --job-name "hello-world" --entry-point "python job.py"
```

このコマンドは次のことを行います：
1. [W&B Launch jobs repository](https://github.com/wandb/launch-jobs) を一時ディレクトリにクローンします。
2. **hello** プロジェクトで **hello-world-git** という名前のジョブを作成します。ジョブはリポジトリのデフォルトブランチの先頭のコミットと関連付けられます。
3. `jobs/hello_world`ディレクトリと`Dockerfile.wandb`からコンテナイメージをビルドします。
4. コンテナを起動し、`python job.py`を実行します。

特定のブランチまたはコミットハッシュからジョブを作成するには、`-g`、`--git-hash`引数を追加します。引数の完全な一覧を見るには、`wandb launch --help`を実行してください。

### リモートURL形式

Launchジョブに関連付けられたgitリモートは、HTTPS URL または SSH URL のいずれかです。URLのタイプはジョブソースコードを取得するプロトコルを決定します。

| リモートURLタイプ | URL形式 | アクセスと認証の要件 |
| ----------------- | -------- | -------------------- |
| https             | `https://github.com/organization/repository.git` | gitリモートで認証するためのユーザー名とパスワード |
| ssh               | `git@github.com:organization/repository.git` | gitリモートで認証するためのsshキー |

ホスティングプロバイダーによって正確なURL形式は異なることに注意してください。`wandb launch --uri` で作成したジョブは、指定された `--uri` に指定された転送プロトコルを使用します。

## コードアーティファクトジョブ

JobsはW&B Artifactsに保存された任意のソースコードから作成できます。アーティファクトとジョブの新しいコードを作成するには、`--uri`または`-u`引数を使用してローカルディレクトリを指定します。

まず、空のディレクトリを作成し、次の内容のPythonスクリプト`main.py`を追加します：

```python
import wandb

with wandb.init() as run:
    run.log({"metric": 0.5})
```

次の内容の`requirements.txt`ファイルを追加します：

```txt
wandb>=0.17.1
```

次のコマンドでディレクトリをコードアーティファクトとしてログし、ジョブを開始します：

```bash
wandb launch --uri . --job-name hello-world-code --project launch-quickstart --entry-point "python main.py"
```

このコマンドは次のことを行います：
1. 現在のディレクトリを`hello-world-code`という名前のコードアーティファクトとしてログします。
2. `launch-quickstart`プロジェクトに`hello-world-code`という名前のジョブを作成します。
3. 現在のディレクトリからLaunchのデフォルトDockerfileを使ってコンテナイメージをビルドします。デフォルトのDockerfileは`requirements.txt`ファイルをインストールし、エントリーポイントを`python main.py`に設定します。

## イメージジョブ

もう一つの方法として、予め作成されたDockerイメージからジョブをビルドすることもできます。これは、MLコードのための既存のビルドシステムを持っている場合や、ハイパーパラメータや異なるインフラストラクチャースケールを使って実験したいが、コードや要件を調整する必要がない場合に便利です。

イメージはDockerレジストリからプルされ、指定されたエントリーポイント（指定されていない場合はデフォルトのエントリーポイント）で実行されます。Dockerイメージからジョブを作成および実行するために、`--docker-image`オプションに完全なイメージタグを渡します。

以下のコマンドを使用して、プレメイドイメージからシンプルなジョブを実行します：

```bash
wandb launch --docker-image "wandb/job_hello_world:main" --project "hello-world"
```

## 自動ジョブ作成

W&Bは、Launchで作成されていない場合でも、トラックされたソースコードを持つジョブを自動的に作成して追跡します。次の条件のいずれかを満たす場合、runsはトラックされたソースコードを持つとみなされます：
- runに関連するgitリモートおよびコミットハッシュがある
- コードアーティファクトをログした（詳細は[`Run.log_code`](../../ref/python/run.md#log_code)を参照）
- `WANDB_DOCKER`環境変数がイメージタグに設定されたDockerコンテナでrunが実行された

LaunchジョブがW&B runによって自動的に作成された場合、GitリモートURLはローカルgitリポジトリから推測されます。

### Launchジョブの名前

デフォルトでは、W&Bは自動的にジョブ名を生成します。名前はジョブの作成方法（GitHub、コードアーティファクト、またはDockerイメージ）に基づいて生成されます。代わりに、環境変数またはW&B Python SDKを使用してLaunchジョブの名前を定義することができます。

次の表は、ジョブソースに基づいてデフォルトで使用される命名規則について説明しています：

| ソース         | 命名規則                                |
| ------------- | --------------------------------------- |
| GitHub        | `job-<git-remote-url>-<path-to-script>` |
| Code artifact | `job-<code-artifact-name>`              |
| Docker image  | `job-<image-name>`                      |

W&B環境変数またはW&B Python SDKを使用してジョブの名前を設定します。

<Tabs
defaultValue="env_var"
values={[
{label: 'Environment variable', value: 'env_var'},
{label: 'W&B Python SDK', value: 'python_sdk'},
]}>
<TabItem value="env_var">

環境変数`WANDB_JOB_NAME`を使って好みのジョブ名を設定します。例えば：

```bash
WANDB_JOB_NAME=awesome-job-name
```

</TabItem>
<TabItem value="python_sdk">

`wandb.Settings`を使用してジョブの名前を定義します。その後、このオブジェクトを`wandb.init`でW&Bを初期化する際に渡します。例えば：

```python
settings = wandb.Settings(job_name="my-job-name")
wandb.init(settings=settings)
```

</TabItem>
</Tabs>

:::note
Dockerイメージジョブの場合、バージョンエイリアスが自動的にジョブのエイリアスとして追加されます。
:::

## コンテナ化

ジョブはコンテナ内で実行されます。イメージジョブはプレビルドされたDockerイメージを使用し、Gitやコードアーティファクトジョブはコンテナビルドステップが必要です。

ジョブのコンテナ化は、`wandb launch`の引数とジョブソースコード内のファイルでカスタマイズできます。

### ビルドコンテキスト

ビルドコンテキストとは、コンテナイメージをビルドするためにDockerデーモンに送信されるファイルとディレクトリのツリーを指します。デフォルトでは、Launchはジョブソースコードのルートをビルドコンテキストとして使用します。サブディレクトリをビルドコンテキストとして指定するには、ジョブを作成および開始するときに`wandb launch`の`--build-context`引数を使用します。

:::tip
`--build-context`引数は、複数のProjectsを持つモノリポを参照するGitジョブで作業する際に特に便利です。サブディレクトリをビルドコンテキストとして指定することで、モノリポ内の特定のProjectのためのコンテナイメージをビルドできます。

公式のW&B Launch jobs repositoryを使用した例については、[上記の例](#git-jobs)を参照してください。
:::

### Dockerfile

DockerfileはDockerイメージをビルドするための指示を含むテキストファイルです。デフォルトでは、Launchは`requirements.txt`ファイルをインストールするデフォルトのDockerfileを使用します。カスタムDockerfileを使用するには、`wandb launch`の`--dockerfile`引数でファイルのパスを指定します。

Dockerfileのパスはビルドコンテキストに対して相対的に指定します。例えば、ビルドコンテキストが`jobs/hello_world`で、Dockerfileが`jobs/hello_world`ディレクトリにある場合、`--dockerfile`引数は`Dockerfile.wandb`に設定されるべきです。公式のW&B Launch jobs repositoryを使用した例については、[上記の例](#git-jobs)を参照してください。

### Requirementsファイル

カスタムDockerfileが提供されていない場合、LaunchはPython依存関係をインストールするビルドコンテキストを検索します。ビルドコンテキストのルートに`requirements.txt`ファイルが見つかった場合、Launchはファイルに記載された依存関係をインストールします。そうでない場合は、`pyproject.toml`ファイルが見つかった場合、Launchは`project.dependencies`セクションから依存関係をインストールします。