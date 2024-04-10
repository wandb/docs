---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# NVIDIA NeMo Inference Microservice Deploy Job

Deploy a model from W&B Artifacts to the NVIDIA NeMo Inference Microservice.

The Launch Job shown in this guide accepts a compatible model artifact from W&B and deploys to a running NIM/Triton server. Launch converts W&B model artifacts to NVIDIA NeMo Model. 

:::info
Deployment time varies by model and machine type. The base Llama2-7b config takes about 1 minute on GCP's `a2-ultragpu-1g`.
:::

## Compatible model types
W&B supports the following model types:
1. [Llama2](https://llama.meta.com/llama2/)
2. [StarCoder](https://github.com/bigcode-project/starcoder)
3. NV-GPT (coming soon)

## Quickstart

1. [Create a launch queue](../launch/add-job-to-queue.md). Within your launch config:
   1. Set `runtime` to `nvidia`
   2. Set `gpus` to the specific GPUs you want to use, or set `gpus` to `all` to use everything.
   ![image](/images/integrations/nim1.png)
2. Launch an agent on your GPU machine. Within your machine's terminal execute:
   ```bash
   wandb launch-agent -e $ENTITY -p $PROJECT -q $QUEUE
   ```
3. Submit the deployment launch job with your desired configs from the [Launch UI](https://wandb.ai/launch)
   1. You can also submit via the CLI:
      ```bash
      wandb launch -d gcr.io/playground-111/deploy-to-nemo:latest \
        -e $ENTITY \
        -p $PROJECT \
        -q $QUEUE \
        -c $CONFIG_JSON_FNAME
      ```
      ![image](/images/integrations/nim2.png)
      
5. You can track the deployment process in the Launch UI.
   ![image](/images/integrations/nim3.png)
   
7. Once complete, you can immediately curl the endpoint to test the model. The model name is always `ensemble`.
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