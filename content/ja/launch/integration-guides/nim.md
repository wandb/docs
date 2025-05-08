---
title: NVIDIA NeMo 推論マイクロサービスデプロイジョブ
menu:
  launch:
    identifier: ja-launch-integration-guides-nim
    parent: launch-integration-guides
url: /ja/guides/integrations/nim
---

モデルアーティファクトを W&B から NVIDIA NeMo Inference Microservice にデプロイします。これを行うには、W&B Launch を使用します。W&B Launch はモデルアーティファクトを NVIDIA NeMo Model に変換し、稼働中の NIM/Triton サーバーにデプロイします。

W&B Launch は現在、以下の互換性のあるモデルタイプを受け入れています:

1. [Llama2](https://llama.meta.com/llama2/)
2. [StarCoder](https://github.com/bigcode-project/starcoder)
3. NV-GPT (近日公開)

{{% alert %}}
デプロイメント時間はモデルとマシンタイプによって異なります。ベースの Llama2-7b 構成は、GCP の `a2-ultragpu-1g` で約1分かかります。
{{% /alert %}}

## クイックスタート

1. [launch キューを作成する]({{< relref path="../create-and-deploy-jobs/add-job-to-queue.md" lang="ja" >}}) まだ持っていない場合は、以下に例としてキュー設定を示します。

   ```yaml
   net: host
   gpus: all # 特定の GPU セットまたは `all` を使用してすべてを使うこともできます
   runtime: nvidia # nvidia コンテナランタイムも必要です
   volume:
     - model-store:/model-store/
   ```

   {{< img src="/images/integrations/nim1.png" alt="image" >}}

2. プロジェクトにこのジョブを作成します:

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
4. 希望する設定でデプロイメントローンチジョブを [Launch UI](https://wandb.ai/launch) から送信します。
   1. CLI から送信することもできます:
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
6. 完了すると、すぐにエンドポイントに curl してモデルをテストできます。モデル名は常に `ensemble` です。
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