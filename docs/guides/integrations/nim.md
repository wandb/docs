---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# NVIDIA NeMo Inference Microservice Deploy Job

W&BからNVIDIA NeMo Inference Microserviceにモデルアーティファクトをデプロイします。この際、W&B Launchを使用します。W&B LaunchはモデルアーティファクトをNVIDIA NeMoモデルに変換し、稼働中のNIM/Tritonサーバーにデプロイします。

W&B Launchは現在、以下の互換性のあるモデルタイプを受け付けています：

1. [Llama2](https://llama.meta.com/llama2/)
2. [StarCoder](https://github.com/bigcode-project/starcoder)
3. NV-GPT (近日公開)


:::info
デプロイメントの時間はモデルとマシンのタイプによって異なります。基本のLlama2-7bコンフィグは、GCPの `a2-ultragpu-1g` で約1分かかります。
:::


## クイックスタート

1. まだ作成していない場合は、[Launchキューを作成](../launch/add-job-to-queue.md)してください。以下にキューの設定例を示します。

   ```yaml
   net: host
   gpus: all # 特定のGPUセットまたは `all` を指定してすべて使用
   runtime: nvidia # また、nvidia container runtimeも必要
   volume:
     - model-store:/model-store/
   ```

   ![image](/images/integrations/nim1.png)

2. プロジェクトにこのジョブを作成します：

   ```bash
   wandb job create -n "deploy-to-nvidia-nemo-inference-microservice" \
      -e $ENTITY \
      -p $PROJECT \
      -E jobs/deploy_to_nvidia_nemo_inference_microservice/job.py \
      -g andrew/nim-updates \
      git https://github.com/wandb/launch-jobs
   ```

3. GPUマシンでエージェントを起動します：
   ```bash
   wandb launch-agent -e $ENTITY -p $PROJECT -q $QUEUE
   ```
4. お好みの設定で[Launch UI](https://wandb.ai/launch)からデプロイメントジョブを提出します。
   1. CLIからも提出できます：
      ```bash
      wandb launch -d gcr.io/playground-111/deploy-to-nemo:latest \
        -e $ENTITY \
        -p $PROJECT \
        -q $QUEUE \
        -c $CONFIG_JSON_FNAME
      ```
      ![image](/images/integrations/nim2.png)
5. Launch UIでデプロイメントプロセスを追跡できます。
   ![image](/images/integrations/nim3.png)
6. 完了したら、エンドポイントにcurlをすぐに実行してモデルをテストできます。モデル名は常に`ensemble`です。
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