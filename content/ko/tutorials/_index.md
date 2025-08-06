---
title: 튜토리얼
description: W&B를 시작하는 데 도움이 되는 인터랙티브 튜토리얼을 이용해보세요.
cascade:
  menu:
    tutorials:
      parent: tutorials
  type: docs
menu:
  tutorials:
    identifier: ko-tutorials-_index
no_list: true
type: docs
---

## 기본 가이드

아래 튜토리얼은 W&B 를 활용한 기계학습 실험 추적, 모델 평가, 하이퍼파라미터 튜닝, 모델 및 데이터셋 버전 관리 등 W&B 의 기본 사용법을 단계별로 안내합니다.

{{< cardpane >}}
  {{< card >}}
    <a href="/tutorials/experiments/">
      <h2 className="card-title">Experiments 추적하기</h2>
    </a>
    <p className="card-content">W&B 를 사용해 기계학습 실험 추적, 모델 체크포인트 관리, 팀원들과의 협업 등을 진행할 수 있습니다.</p>
  {{< /card >}}
  {{< card >}}
    <a href="/tutorials/tables/">
      <h2 className="card-title">예측값 시각화하기</h2>
    </a>
    <p className="card-content">PyTorch 와 MNIST 데이터를 활용하여 트레이닝 과정에서 모델 예측값을 추적, 시각화, 비교할 수 있습니다.</p>
  {{< /card >}}
{{< /cardpane >}}

{{< cardpane >}}
  {{< card >}}
    <a href="/tutorials/sweeps/">
      <h2 className="card-title">하이퍼파라미터 튜닝하기</h2>
    </a>
    <p className="card-content">W&B Sweeps 를 사용하여 학습률, 배치 크기, 히든 레이어 수 등 다양한 하이퍼파라미터 조합을 자동으로 탐색하는 효율적인 방법을 제공합니다.</p>
  {{< /card >}}
  {{< card >}}
    <a href="/tutorials/artifacts/">
      <h2 className="card-title">Models 및 Datasets 추적하기</h2>
    </a>
    <p className="card-content">W&B Artifacts 를 사용해 ML 실험 파이프라인을 손쉽게 추적할 수 있습니다.</p>
  {{< /card >}}
{{< /cardpane >}}


## 인기 ML 프레임워크 튜토리얼
아래 튜토리얼에서 W&B 와 함께 널리 사용되는 ML 프레임워크 및 라이브러리 적용법을 단계별로 배워보세요:

{{< cardpane >}}
  {{< card >}}
    <a href="/tutorials/pytorch">
      <h2 className="card-title">PyTorch</h2>
    </a>
    <p className="card-content">PyTorch 코드에 W&B 를 통합하여 실험 추적을 파이프라인에 손쉽게 추가할 수 있습니다.</p>
  {{< /card >}}
  {{< card >}}
    <a href="/tutorials/huggingface">
      <h2 className="card-title">HuggingFace Transformers</h2>
    </a>
    <p className="card-content">W&B 인테그레이션을 통해 Hugging Face 모델의 성능을 빠르게 시각화할 수 있습니다.</p>
  {{< /card >}}
{{< /cardpane >}}

{{< cardpane >}}
  {{< card >}}
    <a href="/tutorials/tensorflow">
      <h2 className="card-title">Keras</h2>
    </a>
    <p className="card-content">W&B 와 Keras 를 활용하여 실험 추적, 데이터셋 버전 관리, 프로젝트 협업을 진행하세요.</p>
  {{< /card >}}
  {{< card >}}
    <a href="/tutorials/xgboost_sweeps/">
      <h2 className="card-title">XGBoost</h2>
    </a>
    <p className="card-content">W&B 와 XGBoost 를 활용해 실험 추적, 데이터셋 버전 관리, 프로젝트 협업을 할 수 있습니다.</p>
  {{< /card >}}
{{< /cardpane >}}

## 기타 참고 자료

W&B AI Academy 를 방문하여 LLM 트레이닝, 파인튜닝 및 애플리케이션에서의 실제 활용법까지 배워보세요. MLOps, LLMOps 솔루션을 구현하고, 실제 ML 문제를 W&B 코스로 직접 해결해보실 수 있습니다.

- 대형 언어 모델(LLMs)
    - [LLM Engineering: Structured Outputs](https://www.wandb.courses/courses/steering-language-models?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
    - [LLM 기반 앱 만들기](https://www.wandb.courses/courses/building-llm-powered-apps?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
    - [대형 언어 모델 트레이닝 및 파인튜닝](https://www.wandb.courses/courses/training-fine-tuning-LLMs?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
- 효과적인 MLOps
    - [Model CI/CD](https://www.wandb.courses/courses/enterprise-model-management?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
    - [효과적인 MLOps: 모델 개발](https://www.wandb.courses/courses/effective-mlops-model-development?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
    - [기계학습을 위한 CI/CD(GitOps)](https://www.wandb.courses/courses/ci-cd-for-machine-learning?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
    - [프로덕션 ML 파이프라인에서 데이터 검증](https://www.wandb.courses/courses/data-validation-for-machine-learning?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
    - [비즈니스 의사결정 최적화를 위한 기계학습](https://www.wandb.courses/courses/decision-optimization?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
- W&B Models
    - [W&B 101](https://wandb.ai/site/courses/101/?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)
    - [W&B 201: Model Registry](https://www.wandb.courses/courses/201-model-registry?utm_source=wandb_docs&utm_medium=code&utm_campaign=tutorials)