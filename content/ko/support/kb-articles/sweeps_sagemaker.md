---
title: Sweeps 와 SageMaker 를 함께 사용할 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-sweeps_sagemaker
support:
- 스윕
- 'aws

  '
toc_hide: true
type: docs
url: /support/:filename
---

W&B 를 인증하려면 다음 단계를 완료하세요: 내장된 Amazon SageMaker estimator 를 사용하는 경우 `requirements.txt` 파일을 생성하세요. 인증 및 `requirements.txt` 파일 설정에 대한 자세한 내용은 [SageMaker 인테그레이션]({{< relref path="/guides/integrations/sagemaker.md" lang="ko" >}}) 가이드를 참고하세요.

{{% alert %}}
완전한 예제는 [GitHub](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cifar10-sagemaker) 에서 확인할 수 있으며, 추가적인 인사이트는 우리의 [블로그](https://wandb.ai/site/articles/running-sweeps-with-sagemaker) 에서 얻을 수 있습니다.\
SageMaker 와 W&B 를 활용해 Sentiment Analyzer 를 배포하는 방법은 [Deploy Sentiment Analyzer Using SageMaker and W&B 튜토리얼](https://wandb.ai/authors/sagemaker/reports/Deploy-Sentiment-Analyzer-Using-SageMaker-and-W-B--VmlldzoxODA1ODE)에서 확인할 수 있습니다.
{{% /alert %}}