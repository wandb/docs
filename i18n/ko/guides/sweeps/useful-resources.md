---
description: Collection of useful sources for Sweeps.
displayed_sidebar: default
---

# 유용한 자료

<head>
  <title>W&B 스윕에 대해 더 알아보기 위한 자료</title>
</head>

### 학술 논문

Li, Lisha, et al. "[Hyperband: 하이퍼파라미터 최적화를 위한 새로운 밴딧 기반 접근법.](https://arxiv.org/pdf/1603.06560.pdf)" _기계학습 연구 저널_ 18.1 (2017): 6765-6816.

### 스윕 실험

다음 W&B 리포트는 W&B 스윕을 사용하여 하이퍼파라미터 최적화를 탐색하는 프로젝트의 예를 보여줍니다.

* [가뭄 감시 벤치마크 진행 상황](https://wandb.ai/stacey/droughtwatch/reports/Drought-Watch-Benchmark-Progress--Vmlldzo3ODQ3OQ)
  * 설명: 베이스라인 개발 및 가뭄 감시 벤치마크에 대한 제출 탐색.
* [강화학습에서 안전 벌점 튜닝](https://wandb.ai/safelife/benchmark-sweeps/reports/Tuning-Safety-Penalties-in-Reinforcement-Learning---VmlldzoyNjQyODM)
  * 설명: 다양한 부작용 벌점으로 훈련된 에이전트를 세 가지 다른 작업: 패턴 생성, 패턴 제거 및 탐색에서 검토합니다.
* [W&B와 하이퍼파라미터 검색에서의 의미와 노이즈](https://wandb.ai/stacey/pytorch\_intro/reports/Meaning-and-Noise-in-Hyperparameter-Search--Vmlldzo0Mzk5MQ) [Stacey Svetlichnaya](https://wandb.ai/stacey)
  * 설명: 어떻게 신호와 파레이돌리아(상상의 패턴)를 구분할까요? 이 글은 W&B로 가능한 것을 보여주고 추가 탐색을 위한 영감을 제공합니다.
* [Who is Them? 트랜스포머를 이용한 텍스트 중의성 해소](https://wandb.ai/stacey/winograd/reports/Who-is-Them-Text-Disambiguation-with-Transformers--VmlldzoxMDU1NTc)
  * 설명: Hugging Face를 사용하여 자연어 이해 모델 탐색
* [DeepChem: 분자 용해도](https://wandb.ai/stacey/deepchem\_molsol/reports/DeepChem-Molecular-Solubility--VmlldzoxMjQxMjM)
  * 설명: 무작위 포레스트와 딥 넷을 사용하여 분자 구조로부터 화학적 성질 예측.
* [MLOps 소개: 하이퍼파라미터 튜닝](https://wandb.ai/iamleonie/Intro-to-MLOps/reports/Intro-to-MLOps-Hyperparameter-Tuning--VmlldzozMTg2OTk3)
  * 설명: 하이퍼파라미터 최적화가 왜 중요한지 탐구하고 기계학습 모델에 대한 하이퍼파라미터 튜닝을 자동화하기 위한 세 가지 알고리즘을 살펴봅니다.

### 사용 가이드

다음 사용 가이드는 W&B를 사용하여 실제 문제를 해결하는 방법을 보여줍니다:

* [XGBoost와 스윕](https://github.com/wandb/examples/blob/master/examples/wandb-sweeps/sweeps-xgboost/xgboost\_tune.py)
  * 설명: XGBoost를 사용하여 하이퍼파라미터 튜닝을 위한 W&B 스윕 사용 방법.

### 스윕 GitHub 저장소

W&B는 오픈 소스를 지지하며 커뮤니티에서의 기여를 환영합니다. GitHub 저장소는 [https://github.com/wandb/sweeps](https://github.com/wandb/sweeps)에서 찾을 수 있습니다. W&B 오픈 소스 저장소에 기여하는 방법에 대한 정보는 W&B GitHub [기여 지침](https://github.com/wandb/wandb/blob/master/CONTRIBUTING.md)을 참조하세요.