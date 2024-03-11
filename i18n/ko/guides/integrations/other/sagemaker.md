---
description: How to integrate W&B with Amazon SageMaker.
slug: /guides/integrations/sagemaker
displayed_sidebar: default
---

# SageMaker

## SageMaker 인테그레이션

W&B는 [Amazon SageMaker](https://aws.amazon.com/sagemaker/)와 통합되어, 하이퍼파라미터를 자동으로 읽고, 분산된 run을 그룹화하며, 체크포인트에서 run을 재개합니다.

### 인증

W&B는 트레이닝 스크립트와 관련해 `secrets.env`라는 파일을 찾고, `wandb.init()`이 호출될 때 환경으로 로드합니다. `wandb.sagemaker_auth(path="source_dir")`를 호출하여 스크립트에서 `secrets.env` 파일을 생성할 수 있습니다. 이 파일을 `.gitignore`에 추가하는 것을 잊지 마세요!

### 기존 추정기

SageMaker의 사전 구성된 추정기 중 하나를 사용하는 경우, 소스 디렉토리에 wandb를 포함한 `requirements.txt`를 추가해야 합니다.

```
wandb
```

Python 2를 실행하는 추정기를 사용하는 경우, wandb를 설치하기 전에 [wheel](https://pythonwheels.com)에서 psutil을 직접 설치해야 합니다:

```
https://wheels.galaxyproject.org/packages/psutil-5.4.8-cp27-cp27mu-manylinux1_x86_64.whl
wandb
```

:::info
완전한 예제는 [GitHub](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cifar10-sagemaker)에서 확인할 수 있으며, [블로그](https://wandb.ai/site/articles/running-sweeps-with-sagemaker)에서 SageMaker로 스윕을 실행하는 방법에 대해 더 자세히 알아볼 수 있습니다.\
SageMaker와 W&B를 사용하여 감정 분석기를 배포하는 방법에 대한 [튜토리얼](https://wandb.ai/authors/sagemaker/reports/Deploy-Sentiment-Analyzer-Using-SageMaker-and-W-B--VmlldzoxODA1ODE)도 읽어보세요.
:::

:::caution
SageMaker 작업에서 W&B 스윕 에이전트는 SageMaker 인테그레이션이 비활성화되지 않는 한 예상대로 작동하지 않습니다. run에서 SageMaker 인테그레이션을 비활성화하려면 다음과 같이 `wandb.init` 호출을 수정할 수 있습니다:

```
wandb.init(..., settings=wandb.Settings(sagemaker_disable=True))
```
:::