---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Dockerのセットアップ

以下のガイドでは、W&B LaunchをローカルマシンでDockerを使用するための設定方法について説明します。これは、launchエージェント環境とキューのターゲットリソースの両方に適用されます。

同じローカルマシンでジョブを実行し、launchエージェントの環境としてDockerを使用することは、クラスター管理システム（例えばKubernetes）がインストールされていないマシンで計算を行う場合に特に有用です。

また、Dockerキューを使用して強力なワークステーションでワークロードを実行することもできます。

:::tip
このセットアップは、ローカルマシンで実験を行うユーザーや、リモートマシンにSSHで接続してlaunchジョブを送信するユーザーに一般的です。
:::

DockerをW&B Launchと一緒に使用する場合、W&Bはまずイメージをビルドし、そのイメージからコンテナをビルドして実行します。イメージはDockerの`docker run <image-uri>`コマンドでビルドされます。キューの設定は、`docker run`コマンドに渡される追加の引数として解釈されます。

## Dockerキューの設定

launchキューの設定（Dockerターゲットリソース用）は、[`docker run`](../../ref/cli/wandb-docker-run.md) CLIコマンドで定義された同じオプションを受け入れます。

エージェントはキュー設定で定義されたオプションを受け取り、それをlaunchジョブの設定からのオーバーライドとマージして、ターゲットリソース（この場合はローカルマシン）で実行される最終的な`docker run`コマンドを生成します。

2つの構文変換が行われます：

1. 繰り返しオプションはリストとしてキュー設定に定義されます。
2. フラグオプションはBooleanで値が`true`としてキュー設定に定義されます。

例えば、以下のキュー設定：

```json
{
  "env": ["MY_ENV_VAR=value", "MY_EXISTING_ENV_VAR"],
  "volume": "/mnt/datasets:/mnt/datasets",
  "rm": true,
  "gpus": "all"
}
```

は、以下の`docker run`コマンドになります：

```bash
docker run \
  --env MY_ENV_VAR=value \
  --env MY_EXISTING_ENV_VAR \
  --volume "/mnt/datasets:/mnt/datasets" \
  --rm <image-uri> \
  --gpus all
```

ボリュームは文字列のリストまたは単一の文字列として指定できます。複数のボリュームを指定する場合はリストを使用します。

Dockerは、launchエージェント環境から値が割り当てられていない環境変数を自動的に渡します。つまり、launchエージェントが`MY_EXISTING_ENV_VAR`という環境変数を持っている場合、その環境変数はコンテナ内で利用可能です。これは、他の設定キーをキュー設定に公開せずに使用したい場合に便利です。

`docker run`コマンドの`--gpus`フラグは、Dockerコンテナで利用可能なGPUを指定することを可能にします。`gpus`フラグの使用方法については、[Dockerのドキュメント](https://docs.docker.com/config/containers/resource_constraints/#gpu)を参照してください。

:::tip
* Dockerコンテナ内でGPUを使用するには、[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)をインストールしてください。
* コードまたはアーティファクトからソースされたジョブからイメージをビルドする場合、[エージェント](#configure-a-launch-agent-on-a-local-machine)によって使用されるベースイメージをオーバーライドしてNVIDIA Container Toolkitを含めることができます。
  例えば、launchキュー内でベースイメージを`tensorflow/tensorflow:latest-gpu`にオーバーライドできます：

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

W&B CLIを使用してDockerを計算リソースとして使用するキューを作成します：

1. [Launchページ](https://wandb.ai/launch)に移動します。
2. **Create Queue**ボタンをクリックします。
3. キューを作成したい**Entity**を選択します。
4. **Name**フィールドにキューの名前を入力します。
5. **Resource**として**Docker**を選択します。
6. **Configuration**フィールドにDockerキューの設定を定義します。
7. **Create Queue**ボタンをクリックしてキューを作成します。

## ローカルマシンでのlaunchエージェントの設定

`launch-config.yaml`という名前のYAML設定ファイルでlaunchエージェントを設定します。デフォルトでは、W&Bは`~/.config/wandb/launch-config.yaml`で設定ファイルをチェックします。launchエージェントをアクティブにするときに、異なるディレクトリーを指定することもできます。

:::tip
W&B CLIを使用して、launchエージェントのコア設定オプション（設定YAMLファイルの代わりに）を指定できます：最大ジョブ数、W&Bエンティティ、およびlaunchキュー。詳細については、[`wandb launch-agent`](../../ref/cli/wandb-launch-agent.md)コマンドを参照してください。
:::

## コアエージェント設定オプション

以下のタブは、W&B CLIとYAML設定ファイルでコア設定エージェントオプションを指定する方法を示しています：

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

## Dockerイメージビルダー

マシン上のlaunchエージェントはDockerイメージをビルドするように設定できます。デフォルトでは、これらのイメージはマシンのローカルイメージリポジトリに保存されます。launchエージェントがDockerイメージをビルドできるようにするには、launchエージェント設定で`builder`キーを`docker`に設定します：

```yaml title="launch-config.yaml"
builder:
	type: docker
```

エージェントにDockerイメージをビルドさせず、代わりにレジストリからの事前ビルドイメージを使用する場合は、launchエージェント設定で`builder`キーを`noop`に設定します：

```yaml title="launch-config.yaml"
builder:
  type: noop
```

## コンテナレジストリ

LaunchはDockerhub、Google Container Registry、Azure Container Registry、Amazon ECRなどの外部コンテナレジストリを使用します。
異なる環境でジョブを実行したい場合は、エージェントがコンテナレジストリからプルできるように設定します。

launchエージェントとクラウドレジストリを接続する方法の詳細については、[Advanced agent setup](./setup-agent-advanced.md#agent-configuration)ページを参照してください。