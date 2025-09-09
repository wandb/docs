---
title: NVIDIA NeMo 推論マイクロサービスのデプロイジョブ
menu:
  launch:
    identifier: ja-launch-integration-guides-nim
    parent: launch-integration-guides
url: guides/integrations/nim
---

W&B から NVIDIA NeMo Inference Microservice へモデル Artifact をデプロイします。これには W&B Launch を使用します。W&B Launch は model Artifacts を NVIDIA NeMo モデルに変換し、稼働中の NIM/Triton サーバーにデプロイします。

W&B Launch は現在、以下の互換モデルを受け付けています:

1. [Llama2](https://llama.meta.com/llama2/)
2. [StarCoder](https://github.com/bigcode-project/starcoder)
3. NV-GPT（近日対応）

{{% alert %}}
デプロイ時間はモデルとマシンの種類によって異なります。ベースの Llama2-7b 設定では、GCP の `a2-ultragpu-1g` で約 1 分です。
{{% /alert %}}

## クイックスタート

1. まだない場合は [Launch キューを作成]({{< relref path="../create-and-deploy-jobs/add-job-to-queue.md" lang="ja" >}}) してください。以下はキュー設定の例です。

   ```yaml
   net: host
   gpus: all # 特定の GPU セットを指定するか、すべてを使う場合は `all`
   runtime: nvidia # NVIDIA コンテナ ランタイムも必要
   volume:
     - model-store:/model-store/
   ```

   {{< img src="/images/integrations/nim1.png" alt="image" >}}

2. あなたの Project で次のジョブを作成します:

   ```bash
   wandb job create -n "deploy-to-nvidia-nemo-inference-microservice" \
      -e $ENTITY \
      -p $PROJECT \
      -E jobs/deploy_to_nvidia_nemo_inference_microservice/job.py \
      -g andrew/nim-updates \
      git https://github.com/wandb/launch-jobs
   ```

3. GPU マシンでエージェントを起動します:
   ```bash
   wandb launch-agent -e $ENTITY -p $PROJECT -q $QUEUE
   ```
4. [Launch UI](https://wandb.ai/launch) から、希望する設定でデプロイ用の Launch ジョブを送信します。
   1. CLI から送信することもできます:
      ```bash
      wandb launch -d gcr.io/playground-111/deploy-to-nemo:latest \
        -e $ENTITY \
        -p $PROJECT \
        -q $QUEUE \
        -c $CONFIG_JSON_FNAME
      ```
      {{< img src="/images/integrations/nim2.png" alt="image" >}}
5. Launch UI でデプロイの進行状況を追跡できます。
   {{< img src="/images/integrations/nim3.png" alt="image" >}}
6. 完了したら、エンドポイントに curl してすぐにモデルをテストできます。モデル名は常に `ensemble` です。
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