---
description: How to integrate W&B with Amazon SageMaker.
slug: /guides/integrations/sagemaker
displayed_sidebar: default
---

# SageMaker

## SageMaker 통합

W&B는 [Amazon SageMaker](https://aws.amazon.com/sagemaker/)와 통합되어 하이퍼파라미터를 자동으로 읽고, 분산 실행을 그룹화하며, 체크포인트에서 실행을 재개합니다.

### 인증

W&B는 학습 스크립트와 상대적인 `secrets.env`라는 파일을 찾아 `wandb.init()`이 호출될 때 환경으로 로드합니다. `wandb.sagemaker_auth(path="source_dir")`를 호출하여 `secrets.env` 파일을 생성할 수 있습니다. 이 파일을 `.gitignore`에 추가하십시오!

### 기존 추정기

SageMaker의 사전 구성된 추정기 중 하나를 사용하는 경우, wandb가 포함된 `requirements.txt`를 소스 디렉터리에 추가해야 합니다.

```
wandb
```

Python 2를 실행하는 추정기를 사용하는 경우, wandb를 설치하기 전에 [wheel](https://pythonwheels.com)에서 직접 psutil을 설치해야 합니다:

```
https://wheels.galaxyproject.org/packages/psutil-5.4.8-cp27-cp27mu-manylinux1_x86_64.whl
wandb
```

:::info
전체 예제는 [GitHub](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cifar10-sagemaker)에서 확인할 수 있으며, 우리의 [블로그](https://wandb.ai/site/articles/running-sweeps-with-sagemaker)에서 SageMaker와 스윕 실행에 대해 더 읽어볼 수 있습니다.\
또한 SageMaker와 W&B를 사용하여 감정 분석기를 배포하는 방법에 대한 [튜토리얼](https://wandb.ai/authors/sagemaker/reports/Deploy-Sentiment-Analyzer-Using-SageMaker-and-W-B--VmlldzoxODA1ODE)을 읽을 수 있습니다.
:::

:::caution
W&B 스윕 에이전트는 SageMaker 통합이 비활성화되지 않는 한 SageMaker 작업에서 예상대로 동작하지 않습니다. `wandb.init` 호출을 다음과 같이 수정하여 실행에서 SageMaker 통합을 비활성화할 수 있습니다:

```
wandb.init(..., settings=wandb.Settings(sagemaker_disable=True))
```
:::