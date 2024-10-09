---
title: Learn more about sweeps
description: 유용한 Sweeps 자료 모음.
displayed_sidebar: default
---

### Academic papers

Li, Lisha, et al. "[Hyperband: A novel bandit-based approach to hyperparameter optimization.](https://arxiv.org/pdf/1603.06560.pdf)" _The   Journal of Machine Learning Research_ 18.1 (2017): 6765-6816.

### Sweep Experiments

다음의 W&B Reports는 W&B Sweeps를 사용하여 하이퍼파라미터 최적화를 탐구하는 프로젝트의 예시를 보여줍니다.

* [Drought Watch Benchmark Progress](https://wandb.ai/stacey/droughtwatch/reports/Drought-Watch-Benchmark-Progress--Vmlldzo3ODQ3OQ)
  * 설명: Drought Watch 벤치마크에 대한 베이스라인을 개발하고 제출 항목을 탐구합니다.
* [Tuning Safety Penalties in Reinforcement Learning](https://wandb.ai/safelife/benchmark-sweeps/reports/Tuning-Safety-Penalties-in-Reinforcement-Learning---VmlldzoyNjQyODM)
  * 설명: 에이전트가 세 가지 다른 작업(패턴 생성, 패턴 제거, 내비게이션)에서 다양한 부작용 패널티로 훈련된 것을 검사합니다.
* [Meaning and Noise in Hyperparameter Search with W&B](https://wandb.ai/stacey/pytorch_intro/reports/Meaning-and-Noise-in-Hyperparameter-Search--Vmlldzo0Mzk5MQ) [Stacey Svetlichnaya](https://wandb.ai/stacey)
  * 설명: 어떻게 우리는 신호를 상상적 패턴(파레이돌리아)에서 구분할 수 있을까요? 이 기사는 W&B로 가능한 것을 보여주고 추가 탐구를 장려하는 것을 목표로 합니다.
* [Who is Them? Text Disambiguation with Transformers](https://wandb.ai/stacey/winograd/reports/Who-is-Them-Text-Disambiguation-with-Transformers--VmlldzoxMDU1NTc)
  * 설명: Hugging Face를 사용하여 자연어 이해를 위한 모델을 탐구합니다.
* [DeepChem: Molecular Solubility](https://wandb.ai/stacey/deepchem_molsol/reports/DeepChem-Molecular-Solubility--VmlldzoxMjQxMjM)
  * 설명: 랜덤 포레스트와 딥넷을 사용하여 분자 구조로부터 화학적 성질을 예측합니다.
* [Intro to MLOps: Hyperparameter Tuning](https://wandb.ai/iamleonie/Intro-to-MLOps/reports/Intro-to-MLOps-Hyperparameter-Tuning--VmlldzozMTg2OTk3)
  * 설명: 왜 하이퍼파라미터 최적화가 중요한지 탐구하고, 기계학습 모델의 하이퍼파라미터 튜닝을 자동화하는 세 가지 알고리즘을 살펴봅니다.

### selfm-anaged

다음의 사용 가이드는 W&B로 실제 문제를 해결하는 방법을 보여줍니다:

* [Sweeps with XGBoost ](https://github.com/wandb/examples/blob/master/examples/wandb-sweeps/sweeps-xgboost/xgboost_tune.py)
  * 설명: XGBoost를 사용하여 W&B Sweeps로 하이퍼파라미터 튜닝하는 방법.

### Sweep GitHub repository

W&B는 오픈 소스를 지지하며 커뮤니티의 기여를 환영합니다. GitHub 저장소는 [https://github.com/wandb/sweeps](https://github.com/wandb/sweeps)에서 찾을 수 있습니다. W&B 오픈 소스 레포에 기여하는 방법에 대한 정보는 W&B GitHub [기여 가이드라인](https://github.com/wandb/wandb/blob/master/CONTRIBUTING.md)을 참조하세요.