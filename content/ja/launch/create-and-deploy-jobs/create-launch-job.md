---
title: Launch ジョブを作成する
menu:
  launch:
    identifier: ja-launch-create-and-deploy-jobs-create-launch-job
    parent: create-and-deploy-jobs
url: guides/launch/create-launch-job
---

{{< cta-button colabLink="https://colab.research.google.com/drive/1wX0OSVxZJDHRsZaOaOEDx-lLUrO1hHgP" >}}

Launch ジョブは W&B Runs を再現するための設計図です。ジョブは、ワークロードの実行に必要なソースコード、依存関係、入力を取り込む W&B Artifacts です。

`wandb launch` コマンドでジョブを作成して実行します。

{{% alert %}}
実行に送信せずにジョブを作成するには、`wandb job create` コマンドを使用します。詳しくは [command reference docs]({{< relref path="/ref/cli/wandb-job/wandb-job-create.md" lang="ja" >}}) を参照してください。
{{% /alert %}}


## Git ジョブ

W&B Launch では、リモート git リポジトリの特定のコミット、ブランチ、またはタグからコードやその他の追跡対象アセットをクローンして実行する Git ベースのジョブを作成できます。`--uri` または `-u` フラグでコードを含む URI を指定し、必要に応じて `--build-context` フラグでサブディレクトリーを指定します。

次のコマンドで git リポジトリから "hello world" ジョブを実行します:

```bash
wandb launch --uri "https://github.com/wandb/launch-jobs.git" --build-context jobs/hello_world --dockerfile Dockerfile.wandb --project "hello-world" --job-name "hello-world" --entry-point "python job.py"
```

このコマンドは次を実行します:
1. [W&B Launch jobs repository](https://github.com/wandb/launch-jobs) を一時ディレクトリーにクローンします。
2. **hello** Project に **hello-world-git** という名前のジョブを作成します。ジョブは、そのリポジトリのデフォルト ブランチの先頭コミットに関連付けられます。
3. `jobs/hello_world` ディレクトリーと `Dockerfile.wandb` を使ってコンテナ イメージをビルドします。
4. コンテナを起動し、`python job.py` を実行します。

特定のブランチまたはコミット ハッシュからジョブをビルドするには、`-g` または `--git-hash` 引数を追加します。利用可能な引数の一覧は `wandb launch --help` を実行してください。

### リモート URL の形式

Launch ジョブに関連付けられた git リモートは、HTTPS または SSH の URL を使用できます。URL の種類によって、ジョブのソースコードを取得する際のプロトコルが決まります。

| リモート URL の種類 | URL 形式 | アクセスと認証の要件 |
| ----------| ------------------- | ------------------------------------------ |
| https      | `https://github.com/organization/repository.git`  | git リモートの認証に必要なユーザー名とパスワード |
| ssh        | `git@github.com:organization/repository.git` | git リモートの認証に必要な SSH キー |

正確な URL 形式はホスティング プロバイダーによって異なります。`wandb launch --uri` で作成したジョブは、指定した `--uri` に含まれる転送プロトコルを使用します。


## Code artifact ジョブ

W&B Artifact に保存された任意のソースコードからジョブを作成できます。ローカル ディレクトリーを `--uri` または `-u` 引数で指定して、新しい code artifact とジョブを作成します。

まず、空のディレクトリーを作成し、次の内容の `main.py` という名前の Python スクリプトを追加します:

```python
import wandb

with wandb.init() as run:
    run.log({"metric": 0.5})
```

次の内容の `requirements.txt` というファイルを追加します:

```txt
wandb>=0.17.1
```

ディレクトリーを code artifact としてログし、次のコマンドでジョブを起動します:

```bash
wandb launch --uri . --job-name hello-world-code --project launch-quickstart --entry-point "python main.py"
```

上記のコマンドは次を実行します:
1. 現在のディレクトリーを `hello-world-code` という名前の code artifact としてログします。
2. `launch-quickstart` Project に `hello-world-code` という名前のジョブを作成します。
3. 現在のディレクトリーと Launch のデフォルト Dockerfile からコンテナ イメージをビルドします。デフォルトの Dockerfile は `requirements.txt` をインストールし、エントリーポイントを `python main.py` に設定します。

## イメージ ジョブ

別の方法として、あらかじめ用意された Docker イメージを基にジョブを作成できます。これは、ML コード用のビルド システムが既に整っている場合や、ジョブ用のコードや要件を変更する予定はないが、ハイパーパラメーターや異なるインフラストラクチャー規模で実験したい場合に便利です。

イメージは Docker レジストリから取得され、指定したエントリーポイントで実行されます。指定がない場合はデフォルトのエントリーポイントが使用されます。`--docker-image` オプションに完全なイメージ タグを渡して、Docker イメージからジョブを作成・実行します。

プリメイドのイメージからシンプルなジョブを実行するには、次のコマンドを使用します:

```bash
wandb launch --docker-image "wandb/job_hello_world:main" --project "hello-world"           
```


## ジョブの自動作成

Launch で作成されていない Run であっても、ソースコードが追跡されている場合は W&B が自動的にジョブを作成し、追跡します。以下のいずれかを満たす Run は、ソースコードが追跡されていると見なされます:
- その Run に関連付けられた git リモートとコミット ハッシュがある。
- その Run が code artifact をログしている。詳しくは [`Run.log_code`]({{< relref path="/ref/python/sdk/classes/run#log_code" lang="ja" >}}) を参照。
- `WANDB_DOCKER` 環境変数にイメージ タグが設定された Docker コンテナで実行された。

W&B の Run によって Launch ジョブが自動作成された場合、Git リモート URL はローカルの git リポジトリから推測されます。

### Launch ジョブ名

デフォルトでは、W&B が自動でジョブ名を生成します。名前はジョブの作成方法（GitHub、code artifact、または Docker イメージ）に応じて生成されます。代わりに、環境変数または W&B Python SDK で Launch ジョブ名を指定することもできます。

以下の表は、ジョブのソースに基づくデフォルトの命名規則を示します:

| ソース        | 命名規則                       |
| ------------- | --------------------------------------- |
| GitHub        | `job-<git-remote-url>-<path-to-script>` |
| Code artifact | `job-<code-artifact-name>`              |
| Docker image  | `job-<image-name>`                      |

W&B の環境変数または W&B Python SDK を使ってジョブに名前を付ける

{{< tabpane text=true >}}
{{% tab "環境変数" %}}
`WANDB_JOB_NAME` 環境変数に希望するジョブ名を設定します。例:

```bash
WANDB_JOB_NAME=awesome-job-name
```
{{% /tab %}}
{{% tab "W&B Python SDK" %}}
`wandb.Settings` でジョブ名を指定し、`wandb.init` で W&B を初期化する際にこのオブジェクトを渡します。例:

```python
settings = wandb.Settings(job_name="my-job-name")
wandb.init(settings=settings)
```
{{% /tab %}}
{{< /tabpane >}}

{{% alert %}}
Docker イメージのジョブでは、バージョン エイリアスが自動的にジョブのエイリアスとして追加されます。
{{% /alert %}}

## コンテナ化

ジョブはコンテナ内で実行されます。イメージ ジョブは事前構築済みの Docker イメージを使用し、Git と code artifact のジョブではコンテナのビルド工程が必要です。

ジョブのコンテナ化は、`wandb launch` の引数やジョブのソースコード内のファイルでカスタマイズできます。

### ビルド コンテキスト

ビルド コンテキストとは、コンテナ イメージをビルドするために Docker デーモンへ送られるファイルとディレクトリーのツリーを指します。デフォルトでは、Launch はジョブのソースコードのルートをビルド コンテキストとして使用します。サブディレクトリーをビルド コンテキストとして指定するには、ジョブの作成・起動時に `wandb launch` の `--build-context` 引数を使用します。

{{% alert %}}
`--build-context` 引数は、複数の Projects を含むモノレポを参照する Git ジョブで特に有用です。サブディレクトリーをビルド コンテキストとして指定することで、モノレポ内の特定の Project 向けにコンテナ イメージをビルドできます。

[上の例]({{< relref path="#git-jobs" lang="ja" >}}) で、公式の W&B Launch jobs リポジトリに対して `--build-context` 引数を使う方法を確認できます。
{{% /alert %}}

### Dockerfile

Dockerfile は、Docker イメージをビルドするための命令が記述されたテキスト ファイルです。デフォルトでは、Launch は `requirements.txt` をインストールするデフォルトの Dockerfile を使用します。カスタム Dockerfile を使用するには、`wandb launch` の `--dockerfile` 引数でそのファイルのパスを指定します。

Dockerfile のパスはビルド コンテキストからの相対パスで指定します。たとえば、ビルド コンテキストが `jobs/hello_world` で、Dockerfile が `jobs/hello_world` ディレクトリー内にある場合、`--dockerfile` 引数は `Dockerfile.wandb` に設定します。公式の W&B Launch jobs リポジトリで `--dockerfile` 引数を使う方法のデモは、[上の例]({{< relref path="#git-jobs" lang="ja" >}}) を参照してください。

### Requirements ファイル

カスタム Dockerfile が指定されていない場合、Launch はビルド コンテキスト内からインストールすべき Python 依存関係を探索します。ビルド コンテキストのルートに `requirements.txt` が見つかった場合は、そのファイルに記載された依存関係をインストールします。見つからない場合でも、`pyproject.toml` があれば、その `project.dependencies` セクションから依存関係をインストールします。