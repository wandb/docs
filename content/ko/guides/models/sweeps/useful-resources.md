---
title: Learn more about sweeps
description: 스윕에 유용한 소스 모음입니다.
menu:
  default:
    identifier: ko-guides-models-sweeps-useful-resources
    parent: sweeps
---

### 학술 논문

Li, Lisha, 외. "[Hyperband: A novel bandit-based approach to hyperparameter optimization.](https://arxiv.org/pdf/1603.06560.pdf)" _The Journal of Machine Learning Research_ 18.1 (2017): 6765-6816.

### Sweep Experiments

다음 W&B Reports는 W&B Sweeps를 사용한 하이퍼파라미터 최적화 실험의 프로젝트 예제를 보여줍니다.

* [Drought Watch Benchmark Progress](https://wandb.ai/stacey/droughtwatch/reports/Drought-Watch-Benchmark-Progress--Vmlldzo3ODQ3OQ)
  * 설명: Drought Watch 베이스라인을 개발하고 제출을 살펴봅니다.
* [Tuning Safety Penalties in Reinforcement Learning](https://wandb.ai/safelife/benchmark-sweeps/reports/Tuning-Safety-Penalties-in-Reinforcement-Learning---VmlldzoyNjQyODM)
  * 설명: 패턴 생성, 패턴 제거 및 탐색이라는 세 가지 다른 작업에서 다양한 부작용 페널티로 훈련된 에이전트를 살펴봅니다.
* [Meaning and Noise in Hyperparameter Search with W&B](https://wandb.ai/stacey/pytorch_intro/reports/Meaning-and-Noise-in-Hyperparameter-Search--Vmlldzo0Mzk5MQ) [Stacey Svetlichnaya](https://wandb.ai/stacey)
  * 설명: 신호와 파레이돌리아(가상 패턴)를 어떻게 구별할까요? 이 기사는 W&B로 가능한 것을 보여주고 추가 탐구를 장려하는 것을 목표로 합니다.
* [Who is Them? Text Disambiguation with Transformers](https://wandb.ai/stacey/winograd/reports/Who-is-Them-Text-Disambiguation-with-Transformers--VmlldzoxMDU1NTc)
  * 설명: 자연어 이해를 위한 모델을 탐색하기 위해 Hugging Face 사용
* [DeepChem: Molecular Solubility](https://wandb.ai/stacey/deepchem_molsol/reports/DeepChem-Molecular-Solubility--VmlldzoxMjQxMjM)
  * 설명: 랜덤 포레스트 및 딥넷을 사용하여 분자 구조로부터 화학적 속성을 예측합니다.
* [Intro to MLOps: Hyperparameter Tuning](https://wandb.ai/iamleonie/Intro-to-MLOps/reports/Intro-to-MLOps-Hyperparameter-Tuning--VmlldzozMTg2OTk3)
  * 설명: 하이퍼파라미터 최적화가 중요한 이유를 알아보고 기계 학습 모델을 위한 하이퍼파라미터 튜닝을 자동화하는 세 가지 알고리즘을 살펴봅니다.

### selfm-anaged

다음 방법 가이드는 W&B로 실제 문제를 해결하는 방법을 보여줍니다.

* [Sweeps with XGBoost ](https://github.com/wandb/examples/blob/master/examples/wandb-sweeps/sweeps-xgboost/xgboost_tune.py)
  * 설명: XGBoost를 사용하여 하이퍼파라미터 튜닝을 위해 W&B Sweeps를 사용하는 방법.

### Sweep GitHub repository

W&B는 오픈 소스를 옹호하며 커뮤니티의 기여를 환영합니다. GitHub 저장소는 [https://github.com/wandb/sweeps](https://github.com/wandb/sweeps)에서 찾을 수 있습니다. W&B 오픈 소스 레포지토리에 기여하는 방법에 대한 자세한 내용은 W&B GitHub [Contribution guidelines](https://github.com/wandb/wandb/blob/master/CONTRIBUTING.md)를 참조하십시오.
