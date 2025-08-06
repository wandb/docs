---
title: ローンチ ジョブを作成する
menu:
  launch:
    identifier: ja-launch-create-and-deploy-jobs-create-launch-job
    parent: create-and-deploy-jobs
url: guides/launch/create-launch-job
---

{{< cta-button colabLink="https://colab.research.google.com/drive/1wX0OSVxZJDHRsZaOaOEDx-lLUrO1hHgP" >}}

Launch ジョブは W&B Run を再現するための設計図です。ジョブは、ワークロードの実行に必要なソースコード、依存関係、インプットを捉える W&B Artifacts です。

`wandb launch` コマンドを使ってジョブを作成・実行できます。

{{% alert %}}
ジョブを実行せずに作成だけしたい場合は、`wandb job create` コマンドを使ってください。詳しくは[コマンドリファレンス]({{< relref path="/ref/cli/wandb-job/wandb-job-create.md" lang="ja" >}})をご覧ください。
{{% /alert %}}


## Git ジョブ

W&B Launch を使えば、特定の commit、branch、または tag からリモートの git リポジトリ上のコードやその他のトラッキング対象アセットを取得して Git ベースのジョブを作成できます。`--uri` または `-u` フラグでコードが含まれる URI を指定し、必要に応じて `--build-context` フラグでサブディレクトリを指定してください。

以下のコマンドで、Git リポジトリから "hello world" ジョブを実行できます：

```bash
wandb launch --uri "https://github.com/wandb/launch-jobs.git" --build-context jobs/hello_world --dockerfile Dockerfile.wandb --project "hello-world" --job-name "hello-world" --entry-point "python job.py"
```

このコマンドの内容は以下の通りです:
1. [W&B Launch jobs リポジトリ](https://github.com/wandb/launch-jobs)を一時ディレクトリにクローンします。
2. **hello** プロジェクトに **hello-world-git** という名前のジョブを作成します。ジョブは、リポジトリのデフォルトブランチの先頭 commit に紐付けられます。
3. `jobs/hello_world` ディレクトリと `Dockerfile.wandb` からコンテナイメージをビルドします。
4. コンテナを起動して `python job.py` を実行します。

特定の branch や commit hash からジョブをビルドしたい場合は、`-g` や `--git-hash` 引数を追加してください。全ての引数のリストは `wandb launch --help` で確認できます。

### リモートURLの形式

Launch ジョブで使われる git リモートは、HTTPS または SSH のいずれかの URL を利用できます。それぞれの URL 種類によって、コード取得に使われるプロトコルが決まります。

| リモートURLタイプ | URL 形式 | アクセス・認証に必要な情報 |
| -----------| ------------------- | -------------------- |
| https      | `https://github.com/organization/repository.git`  | git リモートに認証するためのユーザー名とパスワード |
| ssh        | `git@github.com:organization/repository.git` | git リモートに認証するための ssh キー |

URL の具体的な書き方はホスティング先によって異なる場合があります。`wandb launch --uri` で作成したジョブは、指定した `--uri` のプロトコルで転送します。


## コード アーティファクトジョブ

W&B Artifact に保存された任意のソースコードからジョブを作成できます。ローカルディレクトリを `--uri` または `-u` 引数で指定すると、新しいコードアーティファクトおよびジョブを作成可能です。

まずは、空のディレクトリを作って `main.py` という Python スクリプトを以下の内容で作成してください：

```python
import wandb

with wandb.init() as run:
    run.log({"metric": 0.5})
```

同じディレクトリに、次の内容の `requirements.txt` を作成します：

```txt
wandb>=0.17.1
```

このディレクトリをコードアーティファクトとしてログし、ジョブを起動するには以下のコマンドを使います：

```bash
wandb launch --uri . --job-name hello-world-code --project launch-quickstart --entry-point "python main.py"
```

上記コマンドの内容は以下の通りです：
1. カレントディレクトリを `hello-world-code` という名前でコードアーティファクトとしてログします。
2. `launch-quickstart` プロジェクトに `hello-world-code` という名前のジョブを作成します。
3. カレントディレクトリと Launch のデフォルト Dockerfile からコンテナイメージをビルドします。デフォルト Dockerfile は `requirements.txt` をインストールし、エントリーポイントに `python main.py` を指定します。

## イメージジョブ

他にも、事前に用意した Docker イメージを利用してジョブを構築することもできます。これは、ML コードのビルドシステムが既に構築されている場合や、コードや依存関係に変更がなく、ハイパーパラメータやインフラ環境を変えて実験したい場合に便利です。

イメージは Docker レジストリから取得され、指定した（あるいはデフォルトの）エントリーポイントで実行されます。`--docker-image` オプションにイメージタグを指定して Docker イメージからジョブを作成・実行できます。

事前作成イメージからシンプルなジョブを実行するには、次のコマンドを使います：

```bash
wandb launch --docker-image "wandb/job_hello_world:main" --project "hello-world"
```


## ジョブの自動作成

W&B では、Launch 以外で作成された Run でも、コードがトラッキングされていれば自動的にジョブが作成・管理されます。Run が「ソースコードをトラッキングしている」とみなされる条件は以下のいずれかです：
- Run に紐付く git remote と commit hash がある
- Run でコードアーティファクトをログしている（[`Run.log_code`]({{< relref path="/ref/python/sdk/classes/run#log_code" lang="ja" >}}) を参照）
- Run が `WANDB_DOCKER` 環境変数にイメージタグを指定した docker コンテナで実行されている

Launch ジョブが W&B Run から自動作成される場合、Git リモートの URL はローカルの git リポジトリから自動的に推定されます。

### Launch ジョブ名

デフォルトでは、W&B がジョブ名を自動生成します。名前は GitHub・コードアーティファクト・Docker イメージのいずれから作成されたかによって変わります。また、環境変数や W&B Python SDK でジョブ名を指定することもできます。

以下の表は、ソースに応じたデフォルトのジョブ命名規則です：

| ソース        | 命名規則                         |
| ------------- | ---------------------------------- |
| GitHub        | `job-<git-remote-url>-<path-to-script>` |
| コードアーティファクト | `job-<code-artifact-name>`         |
| Docker イメージ | `job-<image-name>`                 |

W&B の環境変数、または Python SDK でジョブ名を指定する方法

{{< tabpane text=true >}}
{{% tab "環境変数" %}}
`WANDB_JOB_NAME` 環境変数に、希望のジョブ名を指定してください。例：

```bash
WANDB_JOB_NAME=awesome-job-name
```
{{% /tab %}}
{{% tab "W&B Python SDK" %}}
`wandb.Settings` でジョブ名を指定し、そのオブジェクトを `wandb.init` で渡します。例：

```python
settings = wandb.Settings(job_name="my-job-name")
wandb.init(settings=settings)
```
{{% /tab %}}
{{< /tabpane >}}

{{% alert %}}
Docker イメージジョブの場合、version エイリアスが自動でジョブのエイリアスとして追加されます。
{{% /alert %}}

## コンテナ化

ジョブはコンテナ内で実行されます。イメージジョブは事前構築イメージを、Git ジョブやコードアーティファクトジョブはコンテナビルドステップを必要とします。

ジョブのコンテナ化は、`wandb launch` への引数やジョブソースコード内のファイルでカスタマイズできます。

### ビルドコンテキスト

ビルドコンテキストとは、コンテナイメージをビルドする際に Docker デーモンへ送信するファイル・ディレクトリのツリーのことです。デフォルトでは Launch はジョブソースコードのルートをビルドコンテキストとします。特定のサブディレクトリをビルドコンテキストに指定したい場合は、ジョブ作成時の `wandb launch` で `--build-context` 引数を利用してください。

{{% alert %}}
`--build-context` 引数は、複数プロジェクトが存在するモノリポを参照する Git ジョブで特に便利です。サブディレクトリをビルドコンテキストに指定して、モノリポ内の特定プロジェクト向けにコンテナイメージをビルドできます。

公式 W&B Launch jobs リポジトリでの利用例は[上記の例]({{< relref path="#git-jobs" lang="ja" >}})を参照してください。
{{% /alert %}}

### Dockerfile

Dockerfile は Docker イメージをビルドするための指示をまとめたテキストファイルです。デフォルトでは Launch の標準 Dockerfile で `requirements.txt` をインストールしますが、カスタム Dockerfile を利用したい場合は、`wandb launch` の `--dockerfile` 引数でファイルパスを指定できます。

Dockerfile のパスはビルドコンテキストからの相対パスとなります。例えばビルドコンテキストが `jobs/hello_world` で、そのディレクトリ内に Dockerfile がある場合は、`--dockerfile` に `Dockerfile.wandb` を指定します。詳細は[上記の例]({{< relref path="#git-jobs" lang="ja" >}})を参照してください。

### Requirements ファイル

カスタム Dockerfile を指定しない場合、Launch はビルドコンテキスト内で Python 依存関係ファイルを探します。ビルドコンテキスト直下に `requirements.txt` があれば、その内容をインストールします。なければ `pyproject.toml` が見つかった場合、`project.dependencies` セクションから依存関係をインストールします。