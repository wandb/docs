---
title: Create a launch job
menu:
  launch:
    identifier: ja-launch-create-and-deploy-jobs-create-launch-job
    parent: create-and-deploy-jobs
url: guides/launch/create-launch-job
---

{{< cta-button colabLink="https://colab.research.google.com/drive/1wX0OSVxZJDHRsZaOaOEDx-lLUrO1hHgP" >}}

Launch jobs は、W&B の run を再現するための設計図です。Jobs は、ワークロードの実行に必要なソース コード、依存関係、および入力をキャプチャする W&B Artifacts です。

`wandb launch` コマンドを使用して、jobs を作成および実行します。

{{% alert %}}
実行のために送信せずに job を作成するには、`wandb job create` コマンドを使用します。詳細については、[コマンド リファレンス ドキュメント]({{< relref path="/ref/cli/wandb-job/wandb-job-create.md" lang="ja" >}}) を参照してください。
{{% /alert %}}

## Git jobs

W&B Launch を使用して、コードやその他の追跡対象アセットがリモート git リポジトリの特定のコミット、ブランチ、またはタグからクローンされる Git ベースの job を作成できます。コードを含む URI を指定するには、`--uri` または `-u` フラグを使用し、オプションでサブディレクトリーを指定するには `--build-context` フラグを使用します。

次のコマンドを使用して、git リポジトリから "hello world" job を実行します。

```bash
wandb launch --uri "https://github.com/wandb/launch-jobs.git" --build-context jobs/hello_world --dockerfile Dockerfile.wandb --project "hello-world" --job-name "hello-world" --entry-point "python job.py"
```

このコマンドは次の処理を実行します。
1. [W&B Launch jobs リポジトリ](https://github.com/wandb/launch-jobs)を一時ディレクトリーにクローンします。
2. **hello** プロジェクトに **hello-world-git** という名前の job を作成します。この job は、リポジトリのデフォルト ブランチのヘッドにあるコミットに関連付けられています。
3. `jobs/hello_world` ディレクトリーと `Dockerfile.wandb` からコンテナー イメージを構築します。
4. コンテナーを起動し、`python job.py` を実行します。

特定のブランチまたはコミット ハッシュから job を構築するには、`-g`、`--git-hash` 引数を追加します。引数の完全なリストについては、`wandb launch --help` を実行してください。

### リモート URL 形式

Launch job に関連付けられた git リモートは、HTTPS または SSH URL のいずれかになります。URL タイプによって、job のソース コードの取得に使用されるプロトコルが決まります。

| リモート URL タイプ| URL 形式 | アクセスと認証の要件 |
| ----------| ------------------- | ------------------------------------------ |
| https      | `https://github.com/organization/repository.git`  | git リモートで認証するためのユーザー名とパスワード |
| ssh        | `git@github.com:organization/repository.git` | git リモートで認証するための ssh キー |

正確な URL 形式はホスティング プロバイダーによって異なることに注意してください。`wandb launch --uri` で作成された jobs は、指定された `--uri` で指定された転送プロトコルを使用します。

## Code artifact jobs

Jobs は、W&B Artifact に保存されている任意のソース コードから作成できます。`--uri` または `-u` 引数を持つローカル ディレクトリーを使用して、新しい code artifact と job を作成します。

まず、空のディレクトリーを作成し、次の内容で `main.py` という名前の Python スクリプトを追加します。

```python
import wandb

with wandb.init() as run:
    run.log({"metric": 0.5})
```

次の内容で `requirements.txt` ファイルを追加します。

```txt
wandb>=0.17.1
```

ディレクトリーを code artifact として記録し、次のコマンドで job を起動します。

```bash
wandb launch --uri . --job-name hello-world-code --project launch-quickstart --entry-point "python main.py"
```

上記のコマンドは次の処理を実行します。
1. 現在のディレクトリーを `hello-world-code` という名前の code artifact として記録します。
2. `launch-quickstart` プロジェクトに `hello-world-code` という名前の job を作成します。
3. 現在のディレクトリーと Launch のデフォルトの Dockerfile からコンテナー イメージを構築します。デフォルトの Dockerfile は `requirements.txt` ファイルをインストールし、エントリー ポイントを `python main.py` に設定します。

## Image jobs

または、既製の Docker イメージから jobs を構築することもできます。これは、ML コード用に確立されたビルド システムが既にある場合、または job のコードまたは要件を調整することは想定していないが、ハイパーパラメーターやさまざまなインフラストラクチャ スケールを試したい場合に役立ちます。

イメージは Docker レジストリからプルされ、指定されたエントリー ポイント (指定されていない場合はデフォルトのエントリー ポイント) で実行されます。`--docker-image` オプションに完全なイメージ タグを渡して、Docker イメージから job を作成および実行します。

既製のイメージから単純な job を実行するには、次のコマンドを使用します。

```bash
wandb launch --docker-image "wandb/job_hello_world:main" --project "hello-world"           
```

## 自動 job 作成

W&B は、追跡されたソース コードを持つ run に対して、たとえその run が Launch で作成されていなくても、自動的に job を作成して追跡します。Run は、次の 3 つの条件のいずれかが満たされている場合に、追跡されたソース コードを持つと見なされます。
- Run に関連付けられた git リモートとコミット ハッシュがある
- Run が code artifact を記録した (詳細については、[`Run.log_code`]({{< relref path="/ref/python/run.md#log_code" lang="ja" >}}) を参照)
- Run が `WANDB_DOCKER` 環境変数がイメージ タグに設定された Docker コンテナーで実行された

Launch job が W&B run によって自動的に作成された場合、Git リモート URL はローカル git リポジトリから推測されます。

### Launch job 名

デフォルトでは、W&B は自動的に job 名を生成します。名前は、job の作成方法 (GitHub、code artifact、または Docker イメージ) に応じて生成されます。または、環境変数または W&B Python SDK を使用して Launch job の名前を定義することもできます。

次の表は、job ソースに基づいてデフォルトで使用される job 命名規則を示しています。

| ソース        | 命名規則                       |
| ------------- | --------------------------------------- |
| GitHub        | `job-<git-remote-url>-<path-to-script>` |
| Code artifact | `job-<code-artifact-name>`              |
| Docker image  | `job-<image-name>`                      |

W&B 環境変数または W&B Python SDK で job に名前を付けます。

{{< tabpane text=true >}}
{{% tab "Environment variable" %}}
`WANDB_JOB_NAME` 環境変数を優先 job 名に設定します。次に例を示します。

```bash
WANDB_JOB_NAME=awesome-job-name
```
{{% /tab %}}

{{% tab "W&B Python SDK" %}}
`wandb.Settings` で job の名前を定義します。次に、`wandb.init` で W&B を初期化するときに、このオブジェクトを渡します。次に例を示します。

```python
settings = wandb.Settings(job_name="my-job-name")
wandb.init(settings=settings)
```
{{% /tab %}}
{{< /tabpane >}}

{{% alert %}}
docker イメージ job の場合、バージョン エイリアスは job のエイリアスとして自動的に追加されます。
{{% /alert %}}

## コンテナ化

Jobs はコンテナー内で実行されます。イメージ job は事前に構築された Docker イメージを使用しますが、Git および code artifact job にはコンテナーの構築手順が必要です。

Job のコンテナー化は、`wandb launch` への引数と job のソース コード内のファイルを使用してカスタマイズできます。

### ビルド コンテキスト

ビルド コンテキストという用語は、コンテナー イメージを構築するために Docker デーモンに送信されるファイルとディレクトリーのツリーを指します。デフォルトでは、Launch は job のソース コードのルートをビルド コンテキストとして使用します。サブディレクトリーをビルド コンテキストとして指定するには、job の作成と起動時に `wandb launch` の `--build-context` 引数を使用します。

{{% alert %}}
`--build-context` 引数は、複数のプロジェクトを含むモノレポを参照する Git job を操作する場合に特に役立ちます。サブディレクトリーをビルド コンテキストとして指定することで、モノレポ内の特定のプロジェクトのコンテナー イメージを構築できます。

`--build-context` 引数を公式の W&B Launch jobs リポジトリで使用する方法のデモについては、[上記の例]({{< relref path="#git-jobs" lang="ja" >}}) を参照してください。
{{% /alert %}}

### Dockerfile

Dockerfile は、Docker イメージを構築するための命令を含むテキスト ファイルです。デフォルトでは、Launch は `requirements.txt` ファイルをインストールするデフォルトの Dockerfile を使用します。カスタム Dockerfile を使用するには、`wandb launch` の `--dockerfile` 引数を使用してファイルへのパスを指定します。

Dockerfile パスは、ビルド コンテキストを基準にして指定されます。たとえば、ビルド コンテキストが `jobs/hello_world` で、Dockerfile が `jobs/hello_world` ディレクトリーにある場合、`--dockerfile` 引数は `Dockerfile.wandb` に設定する必要があります。`--dockerfile` 引数を公式の W&B Launch jobs リポジトリで使用する方法のデモについては、[上記の例]({{< relref path="#git-jobs" lang="ja" >}}) を参照してください。

### 要件ファイル

カスタム Dockerfile が指定されていない場合、Launch はインストールする Python 依存関係のビルド コンテキストを検索します。`requirements.txt` ファイルがビルド コンテキストのルートに見つかった場合、Launch はファイルにリストされている依存関係をインストールします。それ以外の場合、`pyproject.toml` ファイルが見つかった場合、Launch は `project.dependencies` セクションから依存関係をインストールします。
