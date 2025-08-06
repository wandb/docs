---
title: Sweeps 에 대해 자세히 알아보기
description: Sweeps에 유용한 자료 모음입니다.
menu:
  default:
    identifier: ko-guides-models-sweeps-useful-resources
    parent: sweeps
---

### 학술 논문

Li, Lisha, 외. "[Hyperband: A novel bandit-based approach to hyperparameter optimization.](https://arxiv.org/pdf/1603.06560.pdf)" _The Journal of Machine Learning Research_ 18.1 (2017): 6765-6816.

### Sweep Experiments

아래 W&B Reports에서는 W&B Sweeps를 활용한 하이퍼파라미터 최적화 실험 사례들을 소개합니다.

* [Drought Watch Benchmark Progress](https://wandb.ai/stacey/droughtwatch/reports/Drought-Watch-Benchmark-Progress--Vmlldzo3ODQ3OQ)
  * 설명: 베이스라인 개발 및 Drought Watch 벤치마크에 제출된 여러 접근법을 탐색합니다.
* [Tuning Safety Penalties in Reinforcement Learning](https://wandb.ai/safelife/benchmark-sweeps/reports/Tuning-Safety-Penalties-in-Reinforcement-Learning---VmlldzoyNjQyODM)
  * 설명: 패턴 생성, 패턴 제거, 네비게이션 세 가지 작업에서 서로 다른 부작용 패널티로 학습된 에이전트들을 비교합니다.
* [Meaning and Noise in Hyperparameter Search with W&B](https://wandb.ai/stacey/pytorch_intro/reports/Meaning-and-Noise-in-Hyperparameter-Search--Vmlldzo0Mzk5MQ) [Stacey Svetlichnaya](https://wandb.ai/stacey)
  * 설명: 신호와 패레일돌리아(가상의 패턴)를 어떻게 구분할 수 있을까요? W&B로 가능한 것들을 선보이며 다양한 탐구를 독려합니다.
* [Who is Them? Text Disambiguation with Transformers](https://wandb.ai/stacey/winograd/reports/Who-is-Them-Text-Disambiguation-with-Transformers--VmlldzoxMDU1NTc)
  * 설명: 자연어 이해를 위한 모델을 Hugging Face로 탐색합니다.
* [DeepChem: Molecular Solubility](https://wandb.ai/stacey/deepchem_molsol/reports/DeepChem-Molecular-Solubility--VmlldzoxMjQxMjM)
  * 설명: 분자 구조로부터 랜덤 포레스트와 딥러닝 모델로 화학적 특성을 예측합니다.
* [Intro to MLOps: Hyperparameter Tuning](https://wandb.ai/iamleonie/Intro-to-MLOps/reports/Intro-to-MLOps-Hyperparameter-Tuning--VmlldzozMTg2OTk3)
  * 설명: 하이퍼파라미터 최적화의 중요성을 살펴보고, 기계학습 모델의 하이퍼파라미터 튜닝을 자동화하는 세 가지 알고리즘을 알아봅니다.

### self-managed

아래 가이드는 W&B를 활용해 실제 문제를 어떻게 해결할 수 있는지 보여줍니다.

* [Sweeps with XGBoost ](https://github.com/wandb/examples/blob/master/examples/wandb-sweeps/sweeps-xgboost/xgboost_tune.py)
  * 설명: XGBoost를 사용하여 W&B Sweeps로 하이퍼파라미터 튜닝하는 방법을 소개합니다.

### Sweep GitHub repository

W&B는 오픈 소스를 지향하며 커뮤니티의 기여를 환영합니다. [W&B Sweeps GitHub repository](https://github.com/wandb/sweeps)에서 코드를 확인할 수 있습니다. 오픈소스 레포지토리에 기여하는 방법은 W&B GitHub의 [Contribution guidelines](https://github.com/wandb/wandb/blob/master/CONTRIBUTING.md)를 참고하세요.