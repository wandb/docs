---
title: Can I use Sweeps and SageMaker?
menu:
  support:
    identifier: ko-support-kb-articles-sweeps_sagemaker
support:
- sweeps
- aws
toc_hide: true
type: docs
url: /ko/support/:filename
---

W&B 인증을 받으려면 다음 단계를 완료하세요. 내장된 Amazon SageMaker estimator를 사용하는 경우 `requirements.txt` 파일을 만드세요. 인증 및 `requirements.txt` 파일 설정에 대한 자세한 내용은 [SageMaker integration]({{< relref path="/guides/integrations/sagemaker.md" lang="ko" >}}) 가이드를 참조하세요.

{{% alert %}}
[GitHub](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cifar10-sagemaker)에서 완전한 예제를 찾아보고, [blog](https://wandb.ai/site/articles/running-sweeps-with-sagemaker)에서 추가 정보를 확인하세요. \
SageMaker 및 W&B를 사용하여 감정 분석기를 배포하는 방법에 대한 [튜토리얼](https://wandb.ai/authors/sagemaker/reports/Deploy-Sentiment-Analyzer-Using-SageMaker-and-W-B--VmlldzoxODA1ODE)에 엑세스하세요.
{{% /alert %}}
