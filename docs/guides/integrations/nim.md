---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# NVIDIA NeMo Inference Microservice Deploy Job

Deploy a model from W&B Artifacts to the NVIDIA NeMo Inference Microservice.

This job accepts a compatible model artifact from W&B and deploys to an running NIM/Triton server. It converts supported models to the `.nemo` format

Deployment time varies by model and machine type. The base Llama2-7b config takes about 1 minute on GCP's `a2-ultragpu-1g`.

## Compatible model types

1. Llama2
2. StarCoder
3. NV-GPT (coming soon)

## User Quickstart

1. Create a queue if you don't have one already.  See an example queue config below.
   1. You can set `gpus` to the specific GPUs you want to use, or `all` to use everything.
   2. Set `runtime` to `nvidia`
   
   ![image](/images/integrations/nim1.png)
2. Launch an agent on your GPU machine:
   ```bash
   wandb launch-agent -e $ENTITY -p $PROJECT -q $QUEUE
   ```
3. Submit the deployment job with your desired configs from the [Launch UI](https://wandb.ai/launch). See `configs/` for examples.
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