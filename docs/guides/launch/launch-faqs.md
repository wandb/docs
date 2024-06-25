---
description: W&B Launch に関するよくある質問への回答。
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Launch FAQs

<head>
  <title>Frequently Asked Questions About Launch</title>
</head>

## Getting Started

### I do not want W&B to build a container for me, can I still use Launch?

Yes. Run the following to launch a pre-built docker image. Replace the items in the `<>` with your information:

```bash
wandb launch -d <docker-image-uri> -q <queue-name> -E <entrypoint>
```

This will build a job when you create a run.

Or you can make a job from an image:

```bash
wandb job create image <image-name> -p <project> -e <entity>
```

### Are there best practices for using Launch effectively?

1. エージェントを開始する前にキューを作成しておくと、エージェントを簡単にキューにポイントさせることができます。これをしないと、エージェントがエラーを出し、キューを追加するまで機能しません。
2. エージェントを起動するためにW&Bサービスアカウントを作成し、それが個々のユーザーアカウントに結び付けられないようにします。
3. `wandb.config` を使用してハイパーパラメーターの読み書きを行い、ジョブの再実行時に上書きできるようにします。 argsparse を使用する場合は、[このガイド](https://docs.wandb.ai/guides/launch/create-launch-job#making-your-code-job-friendly) をチェックしてください。

### I do not like clicking- can I use Launch without going through the UI?

Yes. The standard `wandb` CLI includes a `launch` subcommand that you can use to launch your jobs. For more info, try running

```bash
wandb launch --help
```

### Can Launch automatically provision (and spin down) compute resources for me in the target environment?

これは環境に依存します。SageMakerやVertexではリソースを自動的にプロビジョンできます。Kubernetesでは、必要に応じてリソースを自動的にスピンダウンまたはスピンアップするためにオートスケーラーが使用できます。W&Bのソリューションアーキテクトが、再試行、オートスケーリング、スポットインスタンスノードプールの利用を促進するためのKubernetesインフラストラクチャ設定をお手伝いします。support@wandb.com または共有のSlackチャンネルにお問い合わせください。

### Is `wandb launch -d` or `wandb job create image` uploading a whole docker artifact and not pulling from a registry?

No. The `wandb launch -d` command will not upload to a registry for you. You need to upload your image to a registry yourself. Here are the general steps:

1. イメージをビルドします。
2. イメージをレジストリにプッシュします。

このワークフローは次のようになります:

```bash
docker build -t <repo-url>:<tag> .
docker push <repo-url>:<tag>
wandb launch -d <repo-url>:<tag>
```

ここから、launchエージェントがそのコンテナを指すジョブをスピンアップします。 コンテナレジストリからイメージをプルするためのエージェントアクセスを与える方法の例は、 [Advanced agent setup](./setup-agent-advanced.md#agent-configuration) を参照してください。

Kubernetes の場合、Kubernetes クラスターポッドはプッシュするレジストリへのアクセスが必要です。

### Can I specify a Dockerfile and let W&B build a Docker image for me?
Yes. これにより、頻繁に変更がない多数の要件があるが、コードベースが頻繁に変更される場合に特に役立ちます。

:::important
Dockerfile がマウントを使用するようにフォーマットされていることを確認してください。 詳細は、[Docker Docs ウェブサイトの Mounts ドキュメント](https://docs.docker.com/build/guide/mounts/) を参照してください。
:::

Dockerfile の設定が完了したら、Dockerfile を W&B に渡す方法は次の3つのいずれかです:

- Dockerfile.wandb を使用する
- W&B CLI
- W&B アプリ

<Tabs
  defaultValue="dockerfile"
  values={[
    {label: 'Dockerfile.wandb', value: 'dockerfile'},
    {label: 'W&B CLI', value: 'cli'},
    {label: 'W&B App', value: 'app'},
  ]}>
  <TabItem value="dockerfile">

Include a file called `Dockerfile.wandb` in the same directory as the W&B run’s entrypoint.  W&B will use `Dockerfile.wandb` instead of W&B’s built-in Dockerfile.

  </TabItem>
  <TabItem value="cli">

Provide the `--dockerfile` flag when you call queue a launch job with the [`wandb launch`](../../ref/cli/wandb-launch.md) command:

```bash
wandb launch --dockerfile path/to/Dockerfile
```

  </TabItem>
  <TabItem value="app">

When you add a job to a queue on the W&B App, provide the path to your Dockerfile in the **Overrides** section. More specifically, provide it as a key-value pair where `"dockerfile"` is the key and the value is the path to your Dockerfile.

For example, the following JSON shows how to include a Dockerfile that is within a local directory:

```json title="Launch job W&B App"
{
  "args": [],
  "run_config": {
    "lr": 0,
    "batch_size": 0,
    "epochs": 0
  },
  "entrypoint": [],
  "dockerfile": "./Dockerfile"
}
```

  </TabItem>
</Tabs>

## Permissions and Resources

### How do I control who can push to a queue?

キューはユーザーのチームにスコープされます。キューを作成するときに所有エンティティを定義します。 アクセスを制限するには、チームメンバーシップを変更できます。

### What permissions does the agent require in Kubernetes?

次のKubernetesマニフェストは、`wandb`ネームスペース内に`wandb-launch-agent`という名前の役割を作成します。この役割は、エージェントが`wandb`のネームスペース内でポッド、コンフィグマップ、シークレット、ポッド/ログを作成することを許可します。 `wandb-cluster-role`は、エージェントが選択した任意のネームスペースでポッド、ポッド/ログ、シークレット、ジョブ、およびジョブ/ステータスを作成することを許可します。

### Does Launch support parallelization?  How can I limit the resources consumed by a job?

はい、Launch は複数のGPUおよび複数のノードにわたってジョブのスケーリングをサポートします。 詳細については、[このガイド](https://docs.wandb.ai/tutorials/volcano) をご覧ください。

ジョブ間レベルでは、個々のlaunchエージェントは`max_jobs`パラメーターで設定され、このパラメーターはエージェントが同時に実行できるジョブの数を決定します。 また、エージェントがインフラに接続されている限り、特定のキューに複数のエージェントをポイントすることもできます。

CPU/GPU、メモリ、およびその他の要件は、launchキューまたはジョブのrunレベルで設定できます。 詳細については、Kubernetesでリソース制限を設定したキューの設定方法については、[こちら](https://docs.wandb.ai/guides/launch/kubernetes#queue-configuration)を参照してください。

Sweeps の場合、SDKでキュー設定にブロックを追加できます。

```yaml title="queue config"
  scheduler:
    num_workers: 4
```

これにより、同時に並行して実行されるsweepからのrunの数を制限できます。

### When using Docker queues to run multiple jobs that download the same artifact with `use_artifact`, do we re-download the artifact for every single run of the job, or is there any caching going on under the hood?

キャッシュはなく、それぞれのジョブは独立しています。ただし、キュー/エージェントを構成して共有キャッシュをマウントする方法はいくつかあります。 キュー設定でdocker引き数を使用してこれを実現できます。

特別なケースとして、W&Bアーティファクトキャッシュを永続的なボリュームとしてマウントすることもできます。

### Can you specify secrets for jobs/automations? For instance, an API key which you do not wish to be directly visible to users?

はい。 推奨される方法は：

1. runが作成されるネームスペースにシークレットをバニラk8sシークレットとして追加します。 `kubectl create secret -n <namespace> generic <secret_name> <secret value>` のようなものです。

2. 一度このシークレットが作成されると、runが開始されるときにシークレットを注入するためのキュー設定を指定できます。エンドユーザーはシークレットを見ることができず、クラスタ管理者のみが閲覧できます。

### How can admins restrict what ML engineers have access to modify? For example, changing an image tag may be fine but other job settings may not be.

これは、[queue config templates](./setup-queue-advanced.md) によって制御できます。このテンプレートは、管理者によって定義された制限内で、チームの管理者以外のユーザーが編集できるqueueフィールドを公開します。 キューを作成または編集（公開するフィールドとその制限を定義することを含む）できるのは、チーム管理者のみです。

### How does W&B Launch build images?

ジョブのソースとリソース設定でアクセラレーターベースイメージが指定されているかどうかによって、イメージのビルド手順が異なります。

:::note
キュー設定を指定するとき、またはジョブを送信するとき、キューまたはジョブのリソース設定でベースアクセラレータイメージを提供できます：
```json
{
    "builder": {
        "accelerator": {
            "base_image": "image-name"
        }
    }
}
```
:::

ビルドプロセス中に、提供されたジョブの種類とアクセラレーターベースイメージに応じて、次のアクションが取られます：

|                                                     | pythonをaptでインストール | pythonパッケージをインストール | ユーザーと作業ディレクトリを作成 |  コードをイメージにコピー | エントリーポイントを設定 |
|-----------------------------------------------------|:------------------------:|:-----------------------:|:-------------------------:|:--------------------:|:--------------:|
| gitからソースされたジョブ                            |                          |            X            |             X             |           X          |        X       |
| コードからソースされたジョブ                          |                          |            X            |             X             |           X          |        X       |
| gitからソースされ、アクセラレータイメージを提供       |             X            |            X            |             X             |           X          |        X       |
| コードからソースされ、アクセラレータイメージを提供   |             X            |            X            |             X             |           X          |        X       |
| イメージからソースされたジョブ                       |                          |                         |                           |                      |                |

### What requirements does the accelerator base image have?
アクセラレータを使用するジョブには、必要なアクセラレータコンポーネントがインストールされたアクセラレーターベースイメージを提供できます。提供されるアクセラレータイメージのその他の要件には以下が含まれます：
- Debian互換性 (Launch Dockerfile は python を apt-get で取得するため)
- CPU & GPU ハードウェア命令セットとの互換性 (使用する GPU がサポートする CUDA バージョンを確認してください)
- 提供されたアクセラレータバージョンと ML アルゴリズムでインストールされているパッケージの互換性
- ハードウェアとの互換性を設定するための追加手順を必要とするパッケージのインストール

### How do I make W&B Launch work with Tensorflow on GPU?
GPUでTensorflowを使用するジョブでは、エージェントが実行するコンテナビルドのためにカスタムベースイメージを指定する必要があるかもしれません。この場合、リソース設定の `builder.accelerator.base_image` キーの下にイメージタグを追加します。 例:

```json
{
    "gpus": "all",
    "builder": {
        "accelerator": {
            "base_image": "tensorflow/tensorflow:latest-gpu"
        }
    }
}
```

W&B のバージョン0.15.6以前では、`base_image` の親キーとして `accelerator` の代わりに `cuda` を使用してください。

### Can you use a custom repository for packages when Launch builds the image?

はい。そうするには、以下の行を `requirements.txt` に追加し、`index-url` と `extra-index-url` に渡す値をあなたのものに置き換えます：

```text
----index-url=https://xyz@<your-repo-host> --extra-index-url=https://pypi.org/simple
```

`requirements.txt` はジョブのベースルートで定義する必要があります。

## Automatic run re-queuing on preemption

場合によっては、中断後に再開するジョブを設定すると便利です。たとえば、スポットインスタンスで広範なハイパーパラメータスウィープを実行し、より多くのスポットインスタンスがスピンアップされるとジョブが再開するようにしたい場合、LaunchはKubernetesクラスターでこの設定をサポートできます。

Kubernetesキューにジョブを持つノードがスケジューラによってプリエンプトされた場合、そのジョブはキューの最後に自動的に追加され、後で再開できるようになります。この再開されたrunは元のrunと同じ名前を持ち、UIの同じページからフォローできます。この方法でジョブは最大5回まで自動で再キューに追加されます。

Launch はポッドがスケジューラによってプリエンプトされたかどうかを、ポッドが以下の理由で `DisruptionTarget` という条件を持っているかどうかをチェックして検出します：

- `EvictionByEvictionAPI`
- `PreemptionByScheduler`
- `TerminationByKubelet`

ジョブのコードが再開を許可するように構成されている場合、これにより再キューされたrunは中断された場所から再開できます。そうでない場合、runは再キューされたときに最初から開始されます。 詳細については、再開runに関するガイドを参照してください。

現在、プリエンプトされたノードの自動run再キューからオプトアウトする方法はありません。ただし、UIからrunを削除するか、ノードを直接削除すると、再キューされません。

自動run再キューは現在Kubernetesキューのみで利用可能です。SagemakerとVertexはまだサポートされていません。