---
title: NVIDIA NeMo Inference Microservice Deploy Job
menu:
  launch:
    identifier: ja-launch-integration-guides-nim
    parent: launch-integration-guides
url: guides/integrations/nim
---

W&B から NVIDIA NeMo Inference Microservice へモデルアーティファクトをデプロイします。これには W&B Launch を使用します。W&B Launch はモデルアーティファクトを NVIDIA NeMo モデルへと変換し、稼働中の NIM/Triton サーバーへデプロイします。

W&B Launch が現在対応しているモデルタイプは以下の通りです。

1. [Llama2](https://llama.meta.com/llama2/)
2. [StarCoder](https://github.com/bigcode-project/starcoder)
3. NV-GPT（近日対応予定）

{{% alert %}}
デプロイ時間はモデルやマシンタイプによって異なります。Llama2-7b の基本設定は GCP の `a2-ultragpu-1g` で約1分ほどかかります。
{{% /alert %}}

## クイックスタート

1. まだ作成していない場合は、[ローンチキューを作成]({{< relref path="../create-and-deploy-jobs/add-job-to-queue.md" lang="ja" >}})してください。以下はキュー設定の一例です。

   ```yaml
   net: host
   gpus: all # 使用する GPU を指定するか、`all` ですべて利用可能
   runtime: nvidia # nvidia container runtime も必要です
   volume:
     - model-store:/model-store/
   ```

   {{< img src="/images/integrations/nim1.png" alt="image" >}}

2. プロジェクト内で以下のジョブを作成します：

   ```bash
   wandb job create -n "deploy-to-nvidia-nemo-inference-microservice" \
      -e $ENTITY \
      -p $PROJECT \
      -E jobs/deploy_to_nvidia_nemo_inference_microservice/job.py \
      -g andrew/nim-updates \
      git https://github.com/wandb/launch-jobs
   ```

3. GPU マシン上でエージェントを起動します：
   ```bash
   wandb launch-agent -e $ENTITY -p $PROJECT -q $QUEUE
   ```
4. ご希望の設定で [Launch UI](https://wandb.ai/launch) からデプロイメントローンチジョブを送信します。
   1. CLI 経由でも送信できます：
      ```bash
      wandb launch -d gcr.io/playground-111/deploy-to-nemo:latest \
        -e $ENTITY \
        -p $PROJECT \
        -q $QUEUE \
        -c $CONFIG_JSON_FNAME
      ```
      {{< img src="/images/integrations/nim2.png" alt="image" >}}
5. Launch UI でデプロイメントプロセスの進捗を確認できます。
   {{< img src="/images/integrations/nim3.png" alt="image" >}}
6. 完了後、すぐにエンドポイントへ curl でリクエストしてモデルをテストできます。モデル名は常に `ensemble` です。
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