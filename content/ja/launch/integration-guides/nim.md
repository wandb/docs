---
title: NVIDIA NeMo Inference Microservice Deploy Job
menu:
  launch:
    identifier: ja-launch-integration-guides-nim
    parent: launch-integration-guides
url: guides/integrations/nim
---

W&B から NVIDIA NeMo Inference Microservice にモデル Artifact をデプロイするには、W&B Launch を使用します。W&B Launch は、モデル Artifact を NVIDIA NeMo Model に変換し、実行中の NIM/Triton サーバーにデプロイします。

W&B Launch は現在、以下の互換性のあるモデルタイプを受け入れています。

1. [Llama2](https://llama.meta.com/llama2/)
2. [StarCoder](https://github.com/bigcode-project/starcoder)
3. NV-GPT (近日公開)

{{% alert %}}
デプロイ時間は、モデルとマシンの種類によって異なります。ベースとなる Llama2-7b の構成には、GCP の `a2-ultragpu-1g` で約 1 分かかります。
{{% /alert %}}

## クイックスタート

1. まだ [Launch キューを作成]({{< relref path="../create-and-deploy-jobs/add-job-to-queue.md" lang="ja" >}}) していない場合は作成します。以下のキュー構成の例を参照してください。

   ```yaml
   net: host
   gpus: all # 特定の GPU のセット、またはすべてを使用する場合は `all`
   runtime: nvidia # nvidia container runtime も必要
   volume:
     - model-store:/model-store/
   ```

   {{< img src="/images/integrations/nim1.png" alt="image" >}}

2. プロジェクト でこのジョブを作成します。

   ```bash
   wandb job create -n "deploy-to-nvidia-nemo-inference-microservice" \
      -e $ENTITY \
      -p $PROJECT \
      -E jobs/deploy_to_nvidia_nemo_inference_microservice/job.py \
      -g andrew/nim-updates \
      git https://github.com/wandb/launch-jobs
   ```

3. GPU マシンで エージェント を起動します。
   ```bash
   wandb launch-agent -e $ENTITY -p $PROJECT -q $QUEUE
   ```
4. [Launch UI](https://wandb.ai/launch) から、必要な構成でデプロイメント Launch ジョブを送信します。
   1. CLI から送信することもできます。
      ```bash
      wandb launch -d gcr.io/playground-111/deploy-to-nemo:latest \
        -e $ENTITY \
        -p $PROJECT \
        -q $QUEUE \
        -c $CONFIG_JSON_FNAME
      ```
      {{< img src="/images/integrations/nim2.png" alt="image" >}}
5. デプロイメント プロセス を Launch UI で追跡できます。
   {{< img src="/images/integrations/nim3.png" alt="image" >}}
6. 完了したら、すぐにエンドポイントを curl してモデルをテストできます。モデル名は常に `ensemble` です。
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
