---
title: NVIDIA NeMo Inference Microservice Deploy Job
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

NVIDIA NeMo Inference Microservice로 W&B에서 모델 아티팩트를 배포하십시오. 이를 위해 W&B Launch를 사용하십시오. W&B Launch는 모델 아티팩트를 NVIDIA NeMo Model로 변환하여 실행 중인 NIM/Triton 서버에 배포합니다.

현재 W&B Launch는 다음과 같은 호환 가능한 모델 유형을 지원합니다:

1. [Llama2](https://llama.meta.com/llama2/)
2. [StarCoder](https://github.com/bigcode-project/starcoder)
3. NV-GPT (곧 출시 예정)


:::info
배포 시간은 모델과 머신 유형에 따라 다릅니다. 기본 Llama2-7b 구성은 GCP의 `a2-ultragpu-1g`에서 약 1분이 소요됩니다.
:::


## 퀵스타트

1. [런치 큐를 생성](../launch/add-job-to-queue.md)합니다. 이미 존재하지 않는 경우 아래 예제 큐 구성을 참조하십시오.

   ```yaml
   net: host
   gpus: all # 특정 GPU 세트 또는 `all`을 사용하여 모든 것을 사용 가능
   runtime: nvidia # nvidia 컨테이너 런타임도 필요
   volume:
     - model-store:/model-store/
   ```

   ![image](/images/integrations/nim1.png)

2. 프로젝트에서 이 작업을 생성합니다:

   ```bash
   wandb job create -n "deploy-to-nvidia-nemo-inference-microservice" \
      -e $ENTITY \
      -p $PROJECT \
      -E jobs/deploy_to_nvidia_nemo_inference_microservice/job.py \
      -g andrew/nim-updates \
      git https://github.com/wandb/launch-jobs
   ```

3. GPU 머신에서 에이전트를 실행합니다:
   ```bash
   wandb launch-agent -e $ENTITY -p $PROJECT -q $QUEUE
   ```
4. 원하는 구성으로 [Launch UI](https://wandb.ai/launch)에서 배포 런치 작업을 제출합니다.
   1. CLI를 통해서도 제출할 수 있습니다:
      ```bash
      wandb launch -d gcr.io/playground-111/deploy-to-nemo:latest \
        -e $ENTITY \
        -p $PROJECT \
        -q $QUEUE \
        -c $CONFIG_JSON_FNAME
      ```
      ![image](/images/integrations/nim2.png)
5. Launch UI에서 배포 프로세스를 추적할 수 있습니다.
   ![image](/images/integrations/nim3.png)
6. 완료되면 모델을 테스트하기 위해 즉시 엔드포인트에 curl을 수행할 수 있습니다. 모델 이름은 항상 `ensemble`입니다.
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