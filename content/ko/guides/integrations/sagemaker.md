---
title: SageMaker
description: W&B를 Amazon SageMaker와 연동하는 방법
menu:
  default:
    identifier: ko-guides-integrations-sagemaker
    parent: integrations
weight: 370
---

W&B는 [Amazon SageMaker](https://aws.amazon.com/sagemaker/)와 연동되어 하이퍼파라미터를 자동으로 읽고, 분산된 run들을 그룹화하며, 체크포인트에서 run을 재개할 수 있습니다.

## 인증

W&B는 트레이닝 스크립트와 동일한 위치에 있는 `secrets.env` 파일을 찾아, `wandb.init()`이 호출될 때 해당 파일의 환경 변수를 환경에 로드합니다. 실험을 시작할 때 사용하는 스크립트 내에서 `wandb.sagemaker_auth(path="source_dir")`를 호출하면 `secrets.env` 파일을 생성할 수 있습니다. 이 파일은 꼭 `.gitignore`에 추가해 주세요!

## 기존 Estimator 사용 시

SageMaker에서 사전에 구성된 estimator를 사용하는 경우, wandb가 포함된 `requirements.txt` 파일을 소스 디렉토리에 반드시 추가해야 합니다.

```text
wandb
```

만약 Python 2 환경에서 동작하는 estimator를 사용한다면, wandb 설치 전에 [이 wheel](https://pythonwheels.com)에서 `psutil`을 직접 설치해야 합니다:

```text
https://wheels.galaxyproject.org/packages/psutil-5.4.8-cp27-cp27mu-manylinux1_x86_64.whl
wandb
```

전체 예제 코드는 [GitHub](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cifar10-sagemaker)에서 확인할 수 있으며, 자세한 설명은 우리의 [블로그](https://wandb.ai/site/articles/running-sweeps-with-sagemaker)에서 볼 수 있습니다.

또한, SageMaker와 W&B를 사용해 센티멘트 분석기를 배포하는 방법은 [Deploy Sentiment Analyzer Using SageMaker and W&B 튜토리얼](https://wandb.ai/authors/sagemaker/reports/Deploy-Sentiment-Analyzer-Using-SageMaker-and-W-B--VmlldzoxODA1ODE)에서 안내하고 있습니다.

{{% alert color="secondary" %}}
W&B sweep 에이전트는 SageMaker 연동 기능이 꺼져 있을 때만 SageMaker job에서 정상적으로 동작합니다. `wandb.init` 호출 시 아래와 같이 SageMaker 연동을 비활성화해주세요:

```python
# SageMaker 연동 기능 비활성화
wandb.init(..., settings=wandb.Settings(sagemaker_disable=True))
```
{{% /alert %}}