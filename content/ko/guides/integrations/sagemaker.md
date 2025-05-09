---
title: SageMaker
description: Amazon SageMaker와 W&B를 통합하는 방법.
menu:
  default:
    identifier: ko-guides-integrations-sagemaker
    parent: integrations
weight: 370
---

W&B 는 [Amazon SageMaker](https://aws.amazon.com/sagemaker/) 와 통합되어 하이퍼파라미터를 자동으로 읽고, 분산된 Runs 를 그룹화하며, 체크포인트에서 Runs 를 재개합니다.

## 인증

W&B 는 트레이닝 스크립트와 관련된 `secrets.env` 라는 파일을 찾고 `wandb.init()` 가 호출될 때 해당 파일을 환경에 로드합니다. `secrets.env` 파일은 실험을 시작하는 데 사용하는 스크립트에서 `wandb.sagemaker_auth(path="source_dir")` 를 호출하여 생성할 수 있습니다. 이 파일을 `.gitignore` 에 추가해야 합니다!

## 기존 estimator

SageMaker 의 사전 구성된 estimator 중 하나를 사용하는 경우, wandb 를 포함하는 `requirements.txt` 를 소스 디렉터리에 추가해야 합니다.

```text
wandb
```

Python 2 를 실행하는 estimator 를 사용하는 경우, wandb 를 설치하기 전에 이 [wheel](https://pythonwheels.com) 에서 `psutil` 을 직접 설치해야 합니다.

```text
https://wheels.galaxyproject.org/packages/psutil-5.4.8-cp27-cp27mu-manylinux1_x86_64.whl
wandb
```

[GitHub](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cifar10-sagemaker) 에서 전체 예제를 검토하고, [블로그](https://wandb.ai/site/articles/running-sweeps-with-sagemaker) 에서 자세한 내용을 읽어보세요.

SageMaker 와 W&B 를 사용하여 감성 분석기를 배포하는 방법에 대한 [튜토리얼](https://wandb.ai/authors/sagemaker/reports/Deploy-Sentiment-Analyzer-Using-SageMaker-and-W-B--VmlldzoxODA1ODE) 도 읽어볼 수 있습니다.

{{% alert color="secondary" %}}
W&B 스윕 에이전트는 SageMaker 통합이 꺼져 있는 경우에만 SageMaker 작업에서 예상대로 작동합니다. `wandb.init` 호출을 수정하여 SageMaker 통합을 끕니다.

```python
wandb.init(..., settings=wandb.Settings(sagemaker_disable=True))
```
{{% /alert %}}
