---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Docker のセットアップ

以下のガイドでは、W&B Launch をローカルマシンで使用するための Docker の設定方法について説明します。これは、launch エージェントの環境とキューのターゲットリソースの両方に適用されます。

同じローカルマシンでジョブを実行し、launch エージェントの環境として Docker を使用することは、クラスター管理システム（例：Kubernetes）がインストールされていないマシンに計算リソースがある場合に特に有用です。

また、Docker キューを使用して強力なワークステーションでワークロードを実行することもできます。

:::tip
このセットアップは、ローカルマシンで実験を行うユーザーや、リモートマシンに SSH で接続して launch ジョブを送信するユーザーに一般的です。
:::

W&B Launch で Docker を使用する場合、W&B はまずイメージをビルドし、そのイメージからコンテナをビルドして実行します。イメージは Docker の `docker run <image-uri>` コマンドでビルドされます。キュー設定は、`docker run` コマンドに渡される追加の引数として解釈されます。

## Docker キューの設定

Docker ターゲットリソースの launch キュー設定は、[`docker run`](../../ref/cli/wandb-docker-run.md) CLI コマンドで定義された同じオプションを受け入れます。

エージェントはキュー設定で定義されたオプションを受け取り、launch ジョブの設定からのオーバーライドとマージして、ターゲットリソース（この場合はローカルマシン）で実行される最終的な `docker run` コマンドを生成します。

2つの構文変換が行われます：

1. 繰り返しオプションはリストとしてキュー設定に定義されます。
2. フラグオプションは、値が `true` のブール値としてキュー設定に定義されます。

例えば、以下のキュー設定：

```json
{
  "env": ["MY_ENV_VAR=value", "MY_EXISTING_ENV_VAR"],
  "volume": "/mnt/datasets:/mnt/datasets",
  "rm": true,
  "gpus": "all"
}
```

は、以下の `docker run` コマンドになります：

```bash
docker run \
  --env MY_ENV_VAR=value \
  --env MY_EXISTING_ENV_VAR \
  --volume "/mnt/datasets:/mnt/datasets" \
  --rm <image-uri> \
  --gpus all
```

ボリュームは文字列のリストまたは単一の文字列として指定できます。複数のボリュームを指定する場合はリストを使用します。

Docker は、値が割り当てられていない環境変数を launch エージェント環境から自動的に渡します。つまり、launch エージェントに `MY_EXISTING_ENV_VAR` という環境変数がある場合、その環境変数はコンテナ内で利用可能です。これは、キュー設定に公開せずに他の設定キーを使用したい場合に便利です。

`docker run` コマンドの `--gpus` フラグは、Docker コンテナで利用可能な GPU を指定するために使用します。`gpus` フラグの使用方法については、[Docker のドキュメント](https://docs.docker.com/config/containers/resource_constraints/#gpu)を参照してください。

:::tip
* Docker コンテナ内で GPU を使用するには、[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) をインストールしてください。
* コードまたはアーティファクトソースのジョブからイメージをビルドする場合、[エージェント](#configure-a-launch-agent-on-a-local-machine) によって使用されるベースイメージを NVIDIA Container Toolkit を含むようにオーバーライドできます。
  例えば、launch キュー内でベースイメージを `tensorflow/tensorflow:latest-gpu` にオーバーライドできます：

  ```json
  {
    "builder": {
      "accelerator": {
        "base_image": "tensorflow/tensorflow:latest-gpu"
      }
    }
  }
  ```
:::

## キューの作成

W&B CLI を使用して Docker を計算リソースとして使用するキューを作成します：

1. [Launch ページ](https://wandb.ai/launch)に移動します。
2. **Create Queue** ボタンをクリックします。
3. キューを作成したい **Entity** を選択します。
4. **Name** フィールドにキューの名前を入力します。
5. **Resource** として **Docker** を選択します。
6. **Configuration** フィールドに Docker キュー設定を定義します。
7. **Create Queue** ボタンをクリックしてキューを作成します。

## ローカルマシンでの launch エージェントの設定

`launch-config.yaml` という名前の YAML 設定ファイルで launch エージェントを設定します。デフォルトでは、W&B は `~/.config/wandb/launch-config.yaml` に設定ファイルをチェックします。launch エージェントをアクティブにするときに、異なるディレクトリーを指定することもできます。

:::tip
W&B CLI を使用して、launch エージェントのコア設定オプション（設定 YAML ファイルの代わりに）を指定できます：ジョブの最大数、W&B エンティティ、および launch キュー。詳細は [`wandb launch-agent`](../../ref/cli/wandb-launch-agent.md) コマンドを参照してください。
:::

## コアエージェント設定オプション

以下のタブでは、W&B CLI と YAML 設定ファイルを使用してコア設定エージェントオプションを指定する方法を示します：

<Tabs
defaultValue="CLI"
values={[
{label: 'W&B CLI', value: 'CLI'},
{label: 'Config file', value: 'config'},
]}>
<TabItem value="CLI">

```bash
wandb launch-agent -q <queue-name> --max-jobs <n>
```

  </TabItem>
  <TabItem value="config">

```yaml title="launch-config.yaml"
max_jobs: <n concurrent jobs>
queues:
	- <queue-name>
```

  </TabItem>
</Tabs>

## Docker イメージビルダー

マシン上の launch エージェントは Docker イメージをビルドするように設定できます。デフォルトでは、これらのイメージはマシンのローカルイメージリポジトリに保存されます。launch エージェントが Docker イメージをビルドできるようにするには、launch エージェント設定の `builder` キーを `docker` に設定します：

```yaml title="launch-config.yaml"
builder:
	type: docker
```

エージェントが Docker イメージをビルドせず、レジストリから事前にビルドされたイメージを使用する場合、launch エージェント設定の `builder` キーを `noop` に設定します：

```yaml title="launch-config.yaml"
builder:
  type: noop
```

## コンテナレジストリ

Launch は Dockerhub、Google Container Registry、Azure Container Registry、Amazon ECR などの外部コンテナレジストリを使用します。
異なる環境でジョブを実行したい場合、エージェントがコンテナレジストリからプルできるように設定します。

launch エージェントをクラウドレジストリと接続する方法について詳しくは、[Advanced agent setup](./setup-agent-advanced.md#agent-configuration) ページを参照してください。