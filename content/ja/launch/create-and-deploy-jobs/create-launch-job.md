---
title: ローンチジョブを作成する
menu:
  launch:
    identifier: create-launch-job
    parent: create-and-deploy-jobs
url: guides/launch/create-launch-job
---

{{< cta-button colabLink="https://colab.research.google.com/drive/1wX0OSVxZJDHRsZaOaOEDx-lLUrO1hHgP" >}}

Launch ジョブは、W&B の run を再現するための設計図です。ジョブは W&B Artifacts であり、ワークロードを実行するために必要なソースコード、依存関係、入力を記録します。

`wandb launch` コマンドを使って、ジョブの作成と実行ができます。

{{% alert %}}
ジョブを作成するだけで、すぐに実行しない場合は `wandb job create` コマンドを使用してください。詳細は [コマンドリファレンスドキュメント]({{< relref "/ref/cli/wandb-job/wandb-job-create.md" >}}) をご覧ください。
{{% /alert %}}


## Git ジョブ

W&B Launch では、リモート git リポジトリの特定のコミット・ブランチ・タグからコードやその他追跡対象のアセットをクローンして、Git ベースのジョブを作成できます。`--uri` または `-u` フラグでコードを含む URI を指定し、必要に応じて `--build-context` でサブディレクトリを指定します。

以下のコマンドで、git リポジトリから "hello world" ジョブを実行できます。

```bash
wandb launch --uri "https://github.com/wandb/launch-jobs.git" --build-context jobs/hello_world --dockerfile Dockerfile.wandb --project "hello-world" --job-name "hello-world" --entry-point "python job.py"
```

このコマンドは次の処理を行います：
1. [W&B Launch jobs リポジトリ](https://github.com/wandb/launch-jobs) を一時ディレクトリにクローンします。
2. **hello** プロジェクト内に **hello-world-git** という名前のジョブを作成します。このジョブは、リポジトリのデフォルトブランチの最新コミットに紐づきます。
3. `jobs/hello_world` ディレクトリと `Dockerfile.wandb` からコンテナイメージをビルドします。
4. コンテナを起動し、`python job.py` を実行します。

特定のブランチやコミットハッシュからジョブを作成する場合は、`-g`, `--git-hash` 引数を追加してください。引数の一覧は `wandb launch --help` で確認できます。

### リモート URL 形式

Launch ジョブに関連づける git リモートは、HTTPS または SSH のいずれかの URL を使用できます。URL 種類により、ジョブのソースコード取得時のプロトコルが決まります。

| リモート URL 種類 | URL 形式 | アクセス・認証要件 |
| ---------------- | ------------------- | ----------------------- |
| https            | `https://github.com/organization/repository.git`  | Git リモート認証用のユーザー名とパスワード |
| ssh              | `git@github.com:organization/repository.git` | Git リモート認証用の SSH キー |

なお、URL 形式はホスティングサービスによって異なる場合があります。`wandb launch --uri` で作成されたジョブは、指定された `--uri` の転送プロトコルを使用します。


## コード Artifact ジョブ

W&B Artifact に保存された任意のソースコードからジョブを作成できます。ローカルディレクトリを使って `--uri` または `-u` 引数で新たなコード Artifact とジョブを作成します。

まず、空のディレクトリを用意し、`main.py` という Python スクリプトを以下の内容で追加します。

```python
import wandb

with wandb.init() as run:
    run.log({"metric": 0.5})
```

さらに、`requirements.txt` を次の内容で作成してください。

```txt
wandb>=0.17.1
```

ディレクトリをコード Artifact としてログし、ジョブを起動するには以下のコマンドを使用します。

```bash
wandb launch --uri . --job-name hello-world-code --project launch-quickstart --entry-point "python main.py"
```

このコマンドは次の処理を行います：
1. 現在のディレクトリを `hello-world-code` という名前のコード Artifact としてログします。
2. `launch-quickstart` プロジェクト内に `hello-world-code` という名前のジョブを作成します。
3. 現在のディレクトリと Launch のデフォルト Dockerfile からコンテナイメージを作成します。デフォルトの Dockerfile は `requirements.txt` をインストールし、エントリポイントを `python main.py` に設定します。

## イメージジョブ

また、既存の Docker イメージからジョブをビルドすることも可能です。これは、すでに ML コード用のビルドシステムを用意してある場合や、コードや依存関係を変更せず、ハイパーパラメータやインフラ規模だけを調整したい場合に便利です。

イメージは Docker レジストリから取得し、指定したエントリポイント（指定がない場合はデフォルトのエントリポイント）で実行されます。`--docker-image` オプションに完全なイメージタグを指定して、Docker イメージからジョブを作成・実行できます。

既存イメージからシンプルなジョブを実行するには、下記コマンドを使ってください。

```bash
wandb launch --docker-image "wandb/job_hello_world:main" --project "hello-world"           
```

## ジョブの自動作成

W&B では、追跡されたソースコードを持つ run に対して、自動でジョブを作成・トラッキングします（その run が Launch で作成されていなくても）。下記いずれかの条件を満たす場合、run は「追跡されたソースコードを持つ」とみなされます：
- run に git リモートとコミットハッシュが紐づけられている。
- run でコード Artifact をログしている。詳細は [`Run.log_code`]({{< relref "/ref/python/sdk/classes/run#log_code" >}}) を参照。
- run が `WANDB_DOCKER` 環境変数を含む Docker コンテナ内で実行された場合（イメージタグが指定）

W&B run により Launch ジョブが自動作成される場合、Git リモート URL はローカル git リポジトリから推測されます。

### Launch ジョブ名

デフォルトでは、W&B はジョブ名を自動生成します。名前は GitHub・コード Artifact・Docker イメージなど、作成方法によって決まります。もちろん、環境変数や W&B Python SDK でジョブ名を指定することも可能です。

下の表は、デフォルトで使われるジョブ命名規則です：

| ソース        | 命名規則                               |
| ------------- | --------------------------------------- |
| GitHub        | `job-<git-remote-url>-<path-to-script>` |
| コード Artifact | `job-<code-artifact-name>`              |
| Docker イメージ | `job-<image-name>`                      |

W&B 環境変数または Python SDK でジョブに名前を付ける方法：

{{< tabpane text=true >}}
{{% tab "環境変数" %}}
`WANDB_JOB_NAME` 環境変数に希望のジョブ名を設定します。例：

```bash
WANDB_JOB_NAME=awesome-job-name
```
{{% /tab %}}
{{% tab "W&B Python SDK" %}}
`wandb.Settings` でジョブ名を指定し、`wandb.init` でこの設定オブジェクトを渡します。例：

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

ジョブはコンテナ内で実行されます。イメージジョブでは事前にビルドされた Docker イメージを利用し、Git やコード Artifact のジョブではコンテナイメージ作成のビルドステップが必要です。

ジョブのコンテナ化は、`wandb launch` コマンドの引数やジョブソースコード内のファイルでカスタマイズできます。

### ビルドコンテキスト

「ビルドコンテキスト」とは、Docker イメージビルド時に Docker デーモンに送信されるファイル・ディレクトリのツリーを指します。デフォルトでは、Launch がジョブソースコードのルートをビルドコンテキストとします。特定のサブディレクトリをビルドコンテキストにしたい場合は、`wandb launch` で `--build-context` 引数を指定してください。

{{% alert %}}
`--build-context` 引数は、複数のプロジェクトが含まれるモノレポ環境で Git ジョブを扱う際に特に便利です。サブディレクトリをビルドコンテキストに指定することで、モノレポ内の特定プロジェクト向けのコンテナイメージを作成できます。

公式 W&B Launch jobs リポジトリでの `--build-context` 引数使用例については [上記の例]({{< relref "#git-jobs" >}}) をご参照ください。
{{% /alert %}}

### Dockerfile

Dockerfile は Docker イメージ作成手順を記述したテキストファイルです。Launch ではデフォルトで、`requirements.txt` をインストールする Dockerfile を使用します。カスタム Dockerfile を使いたい場合、`wandb launch` の `--dockerfile` 引数にファイルパスを指定します。

Dockerfile のパスはビルドコンテキストからの相対パスです。たとえば、ビルドコンテキストが `jobs/hello_world` で、Dockerfile が同ディレクトリにある場合、`--dockerfile` には `Dockerfile.wandb` を指定します。利用例は [こちら]({{< relref "#git-jobs" >}}) をご覧ください。

### Requirements ファイル

カスタム Dockerfile が指定されない場合、Launch はビルドコンテキスト内の Python 依存関係を自動で探してインストールします。ビルドコンテキストのルートに `requirements.txt` があれば、その内容をインストールします。なければ `pyproject.toml` の `project.dependencies` セクションからインストールします。