---
title: SageMaker
description: W&B를 Amazon SageMaker와 통합하는 방법.
menu:
  default:
    identifier: ko-guides-integrations-sagemaker
    parent: integrations
weight: 370
---

W&B는 [Amazon SageMaker](https://aws.amazon.com/sagemaker/)와 통합되어 하이퍼파라미터를 자동으로 읽고, 분산된 runs을 그룹화하고, 체크포인트에서 runs을 재개합니다.

## 인증

W&B는 트레이닝 스크립트와 관련된 `secrets.env` 파일을 찾아서 `wandb.init()`이 호출될 때 환경에 로드합니다. `wandb.sagemaker_auth(path="source_dir")`를 사용하여 실험을 시작하는 스크립트에서 `secrets.env` 파일을 생성할 수 있습니다. 이 파일을 `.gitignore`에 추가해야 합니다!

## 기존 estimator

SageMaker의 사전 구성된 estimator 중 하나를 사용하는 경우 wandb를 포함하는 `requirements.txt`를 소스 디렉토리에 추가해야 합니다.

```text
wandb
```

Python 2를 실행하는 estimator를 사용하는 경우 wandb를 설치하기 전에 이 [wheel](https://pythonwheels.com)에서 직접 `psutil`을 설치해야 합니다.

```text
https://wheels.galaxyproject.org/packages/psutil-5.4.8-cp27-cp27mu-manylinux1_x86_64.whl
wandb
```

[GitHub](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cifar10-sagemaker)에서 완전한 예제를 검토하고, [블로그](https://wandb.ai/site/articles/running-sweeps-with-sagemaker)에서 자세한 내용을 읽어보세요.

SageMaker와 W&B를 사용하여 감성 분석기를 배포하는 방법에 대한 [튜토리얼](https://wandb.ai/authors/sagemaker/reports/Deploy-Sentiment-Analyzer-Using-SageMaker-and-W-B--VmlldzoxODA1ODE)을 읽을 수도 있습니다.

{{% alert color="secondary" %}}
SageMaker 통합이 꺼져 있는 경우에만 W&B 스윕 에이전트가 SageMaker 작업에서 예상대로 작동합니다. `wandb.init` 호출을 수정하여 SageMaker 통합을 끕니다.

```python
wandb.init(..., settings=wandb.Settings(sagemaker_disable=True))
```
{{% /alert %}}
