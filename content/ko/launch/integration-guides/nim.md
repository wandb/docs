---
title: NVIDIA NeMo Inference Microservice Deploy Job
menu:
  launch:
    identifier: ko-launch-integration-guides-nim
    parent: launch-integration-guides
url: /ko/guides//integrations/nim
---

W&B의 모델 아티팩트를 NVIDIA NeMo Inference Microservice에 배포합니다. 이를 위해 W&B Launch를 사용합니다. W&B Launch는 모델 아티팩트를 NVIDIA NeMo Model로 변환하고 실행 중인 NIM/Triton 서버에 배포합니다.

W&B Launch는 현재 다음과 같은 호환 가능한 모델 유형을 지원합니다.

1. [Llama2](https://llama.meta.com/llama2/)
2. [StarCoder](https://github.com/bigcode-project/starcoder)
3. NV-GPT (출시 예정)

{{% alert %}}
배포 시간은 모델 및 머신 유형에 따라 다릅니다. 기본 Llama2-7b 구성은 GCP의 `a2-ultragpu-1g`에서 약 1분이 소요됩니다.
{{% /alert %}}

## 퀵스타트

1. [Launch queue 생성]({{< relref path="../create-and-deploy-jobs/add-job-to-queue.md" lang="ko" >}}) 아직 없는 경우. 아래 예제 queue 구성을 참조하십시오.

   ```yaml
   net: host
   gpus: all # can be a specific set of GPUs or `all` to use everything
   runtime: nvidia # also requires nvidia container runtime
   volume:
     - model-store:/model-store/
   ```

   {{< img src="/images/integrations/nim1.png" alt="image" >}}

2. 다음 작업을 프로젝트에 생성합니다.

   ```bash
   wandb job create -n "deploy-to-nvidia-nemo-inference-microservice" \
      -e $ENTITY \
      -p $PROJECT \
      -E jobs/deploy_to_nvidia_nemo_inference_microservice/job.py \
      -g andrew/nim-updates \
      git https://github.com/wandb/launch-jobs
   ```

3. GPU 머신에서 에이전트를 실행합니다.
   ```bash
   wandb launch-agent -e $ENTITY -p $PROJECT -q $QUEUE
   ```
4. [Launch UI](https://wandb.ai/launch)에서 원하는 구성으로 배포 Launch 작업을 제출합니다.
   1. CLI를 통해 제출할 수도 있습니다.
      ```bash
      wandb launch -d gcr.io/playground-111/deploy-to-nemo:latest \
        -e $ENTITY \
        -p $PROJECT \
        -q $QUEUE \
        -c $CONFIG_JSON_FNAME
      ```
      {{< img src="/images/integrations/nim2.png" alt="image" >}}
5. Launch UI에서 배포 프로세스를 추적할 수 있습니다.
   {{< img src="/images/integrations/nim3.png" alt="image" >}}
6. 완료되면 엔드포인트를 즉시 curl하여 모델을 테스트할 수 있습니다. 모델 이름은 항상 `ensemble`입니다.
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
