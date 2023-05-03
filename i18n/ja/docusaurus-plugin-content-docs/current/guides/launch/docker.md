---
description: W&B Launchの入門ガイド。
---

# Dockerでの起動

DockerでW&B Launchを使用する方法を学びましょう。

## Launchエージェントの設定

エージェントを実行する場所でDocker CLIをインストールしておく必要があります。Dockerのインストール方法については、[Dockerのドキュメント](https://docs.docker.com/get-docker/)を参照してください。また、作業を進める前に、Dockerデーモンがお使いのマシンで実行されていることを確認してください。Docker上でエージェントがGPUを利用する必要がある場合は、[NVIDIAコンテナツールキット](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) もインストールしてください。

エージェントのデフォルトの振る舞いは、必要に応じてコンテナビルドを行うために、ローカルホストで `docker build` を実行することです。エージェントは、Dockerキューから実行されるrunsをローカルホストで `docker run` を実行することで実行します。エージェントはこれらのアクションにDocker CLIを使用するため、お使いのマシンで設定されているDocker CLIの設定がエージェントによって使用されます。

## Dockerキュー

Dockerキューを作成するには、[このページ](../launch/create-queue.md)のキュー作成手順に従い、リソースタイプとしてDockerを選択します。

![](/images/launch/create-queue.gif)

受け入れ可能な設定値には、`docker run` コマンドで利用可能なすべての引数が含まれます。詳細については、[リファレンス](https://docs.docker.com/engine/reference/commandline/run)を参照してください。複数回指定できるオプションを処理するには、値をリストに格納します。

```json
{
    "env": [
        "MY_ENV_VAR=value",
        "MY_EXISTING_ENV_VAR"
    ],
    "volume": [
         "/mnt/datasets:/mnt/datasets"
    ]
}
```
:::tip

このキューにジョブを送信してGPUを使用するために、次のリソース設定を追加してください。



```json

{

    "gpus": "all"

}

```



リソース設定の`gpus`キーは、`docker run`の`--gpus`引数に値を渡すために使用されます。この引数は、このキューからのrunsを実行する際に、起動エージェントがどのGPUを使用するかを制御するために使用できます。詳細については、関連する[NVIDIAドキュメント](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html#gpu-enumeration) を参照してください。



リマインダー： Dockerを介してGPUを活用するために、エージェントが実行されているマシンに[NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)もインストールする必要があります。

:::



<!-- TODO: put this in a technical FAQ or in the queue docs -->

GPU上でtensorflowを使用するジョブの場合、runsが適切にGPUを利用できるように、エージェントが行うコンテナビルドのカスタムベースイメージを指定する必要がある場合があります。これは、リソース設定の`builder.cuda.base_image`キーの下にイメージタグを追加することで行うことができます。例えば、次のようにします。



```json

{

    "gpus": "all",

    "builder": {

        "cuda": {

            "base_image": "tensorflow/tensorflow:latest-gpu"

        }

    }

}

```