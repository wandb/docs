---
title: NVIDIA NeMo Inference Microservice Deploy Job（デプロイ ジョブ）
menu:
  launch:
    identifier: nim
    parent: launch-integration-guides
url: guides/integrations/nim
---

W&B のモデルアーティファクトを NVIDIA NeMo Inference Microservice にデプロイします。その際に W&B Launch を使用してください。W&B Launch はモデルアーティファクトを NVIDIA NeMo モデルに変換し、稼働中の NIM/Triton サーバーへデプロイします。

W&B Launch が現在対応しているモデルタイプは以下の通りです：

1. [Llama2](https://llama.meta.com/llama2/)
2. [StarCoder](https://github.com/bigcode-project/starcoder)
3. NV-GPT（近日対応予定）

{{% alert %}}
デプロイ時間はモデルやマシンタイプによって異なります。Llama2-7b の基本構成の場合、GCP の `a2-ultragpu-1g` では約 1 分かかります。
{{% /alert %}}

## クイックスタート

1. まだ Launch キューを作成していない場合は、[Launch キューを作成]({{< relref "../create-and-deploy-jobs/add-job-to-queue.md" >}})してください。以下は例となるキュー設定です。

   ```yaml
   net: host
   gpus: all # 特定の GPU セット、または `all` で全て使用可能
   runtime: nvidia # nvidia コンテナランタイムも必要
   volume:
     - model-store:/model-store/
   ```

   {{< img src="/images/integrations/nim1.png" alt="image" >}}

2. あなたの Project でこのジョブを作成します：

   ```bash
   wandb job create -n "deploy-to-nvidia-nemo-inference-microservice" \
      -e $ENTITY \
      -p $PROJECT \
      -E jobs/deploy_to_nvidia_nemo_inference_microservice/job.py \
      -g andrew/nim-updates \
      git https://github.com/wandb/launch-jobs
   ```

3. GPU マシンで agent を起動します：
   ```bash
   wandb launch-agent -e $ENTITY -p $PROJECT -q $QUEUE
   ```
4. [Launch UI](https://wandb.ai/launch) から希望の設定でデプロイメントローンチジョブを送信します。
   1. CLI からも送信可能です：
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
6. 完了したら、すぐにエンドポイントへ curl してモデルをテストできます。モデル名は常に `ensemble` です。
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