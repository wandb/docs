---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Docker のセットアップ

以下のガイドでは、W&B Launch を使用して、ローカルマシンで Docker を使用するための設定方法について説明します。これは、Launch エージェント環境およびキューのターゲットリソースの両方に適用されます。

同じローカルマシンでジョブを実行し、Launch エージェントの環境として Docker を使用することは、特に Kubernetes などのクラスター管理システムがインストールされていないマシンで計算を行う場合に便利です。

強力なワークステーション上でワークロードを実行するために Docker キューを使用することもできます。

:::tip
このセットアップは、ローカルマシンで experiments を実行するユーザーや、リモートマシンに SSH でアクセスして launch ジョブを送信するユーザーによく利用されます。
:::

Docker と W&B Launch を使用すると、最初にイメージがビルドされ、そのイメージからコンテナがビルドされて実行されます。イメージは Docker の `docker run <image-uri>` コマンドでビルドされます。キューの設定は、`docker run` コマンドに渡される追加の引数として解釈されます。

## Docker キューの設定

Docker ターゲットリソース用の launch キューの設定は、[`docker run`](../../ref/cli/wandb-docker-run.md) CLI コマンドで定義されているのと同じオプションを受け入れます。

エージェントは、キュー設定で定義されたオプションを受け取ります。このエージェントは、受け取ったオプションを launch ジョブの設定によるオーバーライドと統合して、ターゲットリソース（この場合はローカルマシン）上で実行される最終的な `docker run` コマンドを生成します。

以下の2つの構文変換が行われます：

1. 繰り返しのオプションは、リストとしてキュー設定に定義されます。
2. フラグオプションは、値が `true` のブール値としてキュー設定に定義されます。

例えば、以下のキュー設定:

```json
{
  "env": ["MY_ENV_VAR=value", "MY_EXISTING_ENV_VAR"],
  "volume": "/mnt/datasets:/mnt/datasets",
  "rm": true,
  "gpus": "all"
}
```

生成される `docker run` コマンドは以下の通りです：

```bash
docker run \
  --env MY_ENV_VAR=value \
  --env MY_EXISTING_ENV_VAR \
  --volume "/mnt/datasets:/mnt/datasets" \
  --rm <image-uri> \
  --gpus all
```

ボリュームは、文字列のリストまたは単一の文字列として指定できます。複数のボリュームを指定する場合は、リストを使用してください。

Docker は、値が割り当てられていない環境変数を自動的に Launch エージェントの環境から渡します。これにより、エージェントに `MY_EXISTING_ENV_VAR` という環境変数がある場合、その環境変数はコンテナ内で利用可能になります。これは、キュー設定に公開せずに他の設定キーを使用したい場合に便利です。

`docker run` コマンドの `--gpus` フラグを使用して、Docker コンテナで利用可能な GPU を指定できます。`gpus` フラグの使用方法については、[Docker ドキュメント](https://docs.docker.com/config/containers/resource_constraints/#gpu) を参照してください。

:::tip
* Docker コンテナ内で GPU を使用するには、[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) をインストールしてください。
* コードまたはアーティファクトソースのジョブからイメージをビルドする場合、エージェントのベースイメージをオーバーライドして NVIDIA Container Toolkit を含めることができます。例えば、launch キュー内でベースイメージを `tensorflow/tensorflow:latest-gpu` にオーバーライドできます：

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

W&B CLI を使用して、Docker を計算リソースとして使用するキューを作成します：

1. [Launch ページ](https://wandb.ai/launch) に移動します。
2. **Create Queue** ボタンをクリックします。
3. キューを作成したい **Entity** を選択します。
4. **Name** フィールドにキューの名前を入力します。
5. **Resource** として **Docker** を選択します。
6. **Configuration** フィールドに Docker キュー設定を定義します。
7. **Create Queue** ボタンをクリックしてキューを作成します。

## ローカルマシンで launch エージェントを設定

Launch エージェントを `launch-config.yaml` という名前の YAML 設定ファイルで設定します。デフォルトでは、W&B は `~/.config/wandb/launch-config.yaml` に設定ファイルをチェックします。Launch エージェントを起動する際に、別のディレクトリーを指定することもできます。

:::tip
設定 YAML ファイルの代わりに W&B CLI を使用して、launch エージェントのコア設定オプション（ジョブの最大数、W&B entity、launch キュー）を指定することができます。詳細については、[`wandb launch-agent`](../../ref/cli/wandb-launch-agent.md) コマンドを参照してください。
:::

## コアエージェント設定オプション

以下のタブで、W&B CLI と YAML 設定ファイルを使用してコアエージェント設定オプションを指定する方法を示しています：

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

マシンの launch エージェントは、Docker イメージをビルドするように設定できます。デフォルトでは、これらのイメージはマシンのローカルイメージリポジトリに保存されます。Launch エージェントが Docker イメージをビルドできるようにするには、launch エージェント設定の `builder` キーを `docker` に設定します：

```yaml title="launch-config.yaml"
builder:
	type: docker
```

エージェントが Docker イメージをビルドせず、代わりにレジストリからプリビルトイメージを使用したい場合は、launch エージェント設定の `builder` キーを `noop` に設定します：

```yaml title="launch-config.yaml"
builder:
  type: noop
```

## コンテナレジストリ

Launch は Dockerhub、Google Container Registry、Azure Container Registry、Amazon ECR といった外部コンテナレジストリを使用します。
ビルドした環境とは異なる環境でジョブを実行したい場合、エージェントがコンテナレジストリからプルできるように設定してください。

Launch エージェントをクラウドレジストリと接続する方法については、[高度なエージェント設定](./setup-agent-advanced.md#agent-configuration)ページを参照してください。