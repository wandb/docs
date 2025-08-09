---
title: NVIDIA NeMo Inference Microservice Deploy Job
menu:
  launch:
    identifier: ko-launch-integration-guides-nim
    parent: launch-integration-guides
url: guides/integrations/nim
---

W&B에서 모델 아티팩트를 NVIDIA NeMo Inference Microservice로 배포하세요. 이를 위해서는 W&B Launch를 사용합니다. W&B Launch는 모델 아티팩트를 NVIDIA NeMo Model로 변환한 뒤, 실행 중인 NIM/Triton 서버에 배포합니다.

현재 W&B Launch에서 지원하는 모델 타입은 다음과 같습니다:

1. [Llama2](https://llama.meta.com/llama2/)
2. [StarCoder](https://github.com/bigcode-project/starcoder)
3. NV-GPT (곧 지원 예정)

{{% alert %}}
배포 시간은 모델 및 머신 종류에 따라 다릅니다. GCP의 `a2-ultragpu-1g`에서 Llama2-7b 기본 설정은 약 1분 정도 소요됩니다.
{{% /alert %}}

## 퀵스타트

1. 아직 없다면, [런치 큐 생성]({{< relref path="../create-and-deploy-jobs/add-job-to-queue.md" lang="ko" >}})을 진행하세요. 아래는 예시 큐 설정입니다.

   ```yaml
   net: host
   gpus: all # 특정 GPU 또는 `all` 전체 사용 가능
   runtime: nvidia # nvidia 컨테이너 런타임 필요
   volume:
     - model-store:/model-store/
   ```

   {{< img src="/images/integrations/nim1.png" alt="image" >}}

2. 아래 job을 프로젝트에 생성하세요:

   ```bash
   wandb job create -n "deploy-to-nvidia-nemo-inference-microservice" \
      -e $ENTITY \
      -p $PROJECT \
      -E jobs/deploy_to_nvidia_nemo_inference_microservice/job.py \
      -g andrew/nim-updates \
      git https://github.com/wandb/launch-jobs
   ```

3. GPU 머신에서 에이전트를 실행하세요:
   ```bash
   wandb launch-agent -e $ENTITY -p $PROJECT -q $QUEUE
   ```
4. [Launch UI](https://wandb.ai/launch)에서 원하는 설정으로 배포 런치 job을 제출하세요.
   1. CLI로도 제출할 수 있습니다:
      ```bash
      wandb launch -d gcr.io/playground-111/deploy-to-nemo:latest \
        -e $ENTITY \
        -p $PROJECT \
        -q $QUEUE \
        -c $CONFIG_JSON_FNAME
      ```
      {{< img src="/images/integrations/nim2.png" alt="image" >}}
5. Launch UI에서 배포 프로세스를 실시간으로 확인할 수 있습니다.
   {{< img src="/images/integrations/nim3.png" alt="image" >}}
6. 배포가 완료되면, 즉시 엔드포인트에 curl 명령어로 접근해 모델을 테스트할 수 있습니다. 모델 이름은 항상 `ensemble`입니다.
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