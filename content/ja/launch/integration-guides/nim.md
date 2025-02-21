---
title: NVIDIA NeMo Inference Microservice Deploy Job
menu:
  launch:
    identifier: ja-launch-integration-guides-nim
    parent: launch-integration-guides
url: guides/integrations/nim
---

W&B から NVIDIA NeMo Inference Microservice にモデルアーティファクトをデプロイします。このためには、W&B Launch を使用します。W&B Launch はモデルアーティファクトを NVIDIA NeMo Model に変換し、動作中の NIM/Triton サーバーにデプロイします。

W&B Launch は現在、以下の互換性のあるモデルタイプを受け付けています:

1. [Llama2](https://llama.meta.com/llama2/)
2. [StarCoder](https://github.com/bigcode-project/starcoder)
3. NV-GPT (近日公開)

{{% alert %}}
デプロイ時間は、モデルとマシンタイプによって異なります。基本的な Llama2-7b の構成は、GCP の `a2-ultragpu-1g` で約 1 分かかります。
{{% /alert %}}

## クイックスタート

1. まだお持ちでない場合は、[launch キューを作成]({{< relref path="../create-and-deploy-jobs/add-job-to-queue.md" lang="ja" >}})します。以下に例としてキューの設定を示します。

   ```yaml
   net: host
   gpus: all # 特定の GPU セットや `all` を指定して全てを使用可能
   runtime: nvidia # nvidia コンテナランタイムも必要
   volume:
     - model-store:/model-store/
   ```

   {{< img src="/images/integrations/nim1.png" alt="image" >}}

2. プロジェクト内でこのジョブを作成します:

   ```bash
   wandb job create -n "deploy-to-nvidia-nemo-inference-microservice" \
      -e $ENTITY \
      -p $PROJECT \
      -E jobs/deploy_to_nvidia_nemo_inference_microservice/job.py \
      -g andrew/nim-updates \
      git https://github.com/wandb/launch-jobs
   ```

3.  GPU マシンでエージェントを起動します:
   ```bash
   wandb launch-agent -e $ENTITY -p $PROJECT -q $QUEUE
   ```
4. Launch UI から希望する設定でデプロイメントランチジョブを送信します。
   1. CLI 経由でも送信可能です:
      ```bash
      wandb launch -d gcr.io/playground-111/deploy-to-nemo:latest \
        -e $ENTITY \
        -p $PROJECT \
        -q $QUEUE \
        -c $CONFIG_JSON_FNAME
      ```
      {{< img src="/images/integrations/nim2.png" alt="image" >}}
5. Launch UI でデプロイメントプロセスを追跡できます。
   {{< img src="/images/integrations/nim3.png" alt="image" >}}
6. 完了したら、エンドポイントをすぐにカールしてモデルをテストできます。モデル名は常に `ensemble` です。
   ```bash
    #!/bin/bash
    curl -X POST "http://0.0.0.0:9999/v1/completions" \
        -H "accept: application/json" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "ensemble",
            "prompt": "Tell me a joke",
            "max_tokens": 256,
            "temperature": 0.5,
            "n": 1,
            "stream": false,
            "stop": "string",
            "frequency_penalty": 0.0
            }'
   ```