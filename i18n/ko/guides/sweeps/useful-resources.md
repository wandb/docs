---
description: Collection of useful sources for Sweeps.
displayed_sidebar: default
---

# 유용한 자료

<head>
  <title>W&B 스윕에 대해 더 알아보기 위한 자료</title>
</head>

### 학술 논문

Li, Lisha, et al. "[Hyperband: 하이퍼파라미터 최적화를 위한 새로운 밴딧 기반 접근법.](https://arxiv.org/pdf/1603.06560.pdf)" _머신 러닝 연구 저널_ 18.1 (2017): 6765-6816.

### 스윕 실험

다음 W&B 리포트들은 W&B 스윕을 이용한 하이퍼파라미터 최적화를 탐구하는 프로젝트의 예시를 보여줍니다.

* [가뭄 감시 벤치마크 진행 상황](https://wandb.ai/stacey/droughtwatch/reports/Drought-Watch-Benchmark-Progress--Vmlldzo3ODQ3OQ)
  * 설명: 가뭄 감시 벤치마크에 대한 기준 개발 및 제출 탐색.
* [강화 학습에서 안전 벌칙 조정](https://wandb.ai/safelife/benchmark-sweeps/reports/Tuning-Safety-Penalties-in-Reinforcement-Learning---VmlldzoyNjQyODM)
  * 설명: 다른 부작용 벌칙으로 훈련된 에이전트를 패턴 생성, 패턴 제거 및 탐색 세 가지 다른 작업에서 검사합니다.
* [W&B에서 하이퍼파라미터 검색의 의미와 노이즈](https://wandb.ai/stacey/pytorch\_intro/reports/Meaning-and-Noise-in-Hyperparameter-Search--Vmlldzo0Mzk5MQ) [Stacey Svetlichnaya](https://wandb.ai/stacey)
  * 설명: 신호와 허상(상상 속 패턴)을 어떻게 구분할까요? 이 글은 W&B로 가능한 것을 보여주며 추가 탐구를 위한 영감을 주고자 합니다.
* [그들은 누구인가? 트랜스포머를 이용한 텍스트 명확화](https://wandb.ai/stacey/winograd/reports/Who-is-Them-Text-Disambiguation-with-Transformers--VmlldzoxMDU1NTc)
  * 설명: Hugging Face를 이용하여 자연어 이해를 위한 모델 탐색
* [DeepChem: 분자 용해도](https://wandb.ai/stacey/deepchem\_molsol/reports/DeepChem-Molecular-Solubility--VmlldzoxMjQxMjM)
  * 설명: 분자 구조로부터 화학적 성질을 랜덤 포레스트와 딥 네트를 사용하여 예측.
* [MLOps 입문: 하이퍼파라미터 튜닝](https://wandb.ai/iamleonie/Intro-to-MLOps/reports/Intro-to-MLOps-Hyperparameter-Tuning--VmlldzozMTg2OTk3)
  * 설명: 하이퍼파라미터 최적화가 왜 중요한지 탐구하고 기계 학습 모델의 하이퍼파라미터 튜닝을 자동화하기 위한 세 가지 알고리즘을 살펴봅니다.

### 사용 방법 가이드

다음 사용 방법 가이드는 W&B를 사용하여 실제 문제를 해결하는 방법을 보여줍니다:

* [XGBoost와 함께하는 스윕](https://github.com/wandb/examples/blob/master/examples/wandb-sweeps/sweeps-xgboost/xgboost\_tune.py)
  * 설명: XGBoost를 사용하여 하이퍼파라미터 튜닝을 위해 W&B 스윕을 사용하는 방법.

### 스윕 GitHub 저장소

W&B는 오픈 소스를 지지하며 커뮤니티로부터의 기여를 환영합니다. GitHub 저장소는 [https://github.com/wandb/sweeps](https://github.com/wandb/sweeps)에서 찾을 수 있습니다. W&B 오픈 소스 저장소에 기여하는 방법에 대한 정보는 W&B GitHub [기여 가이드라인](https://github.com/wandb/wandb/blob/master/CONTRIBUTING.md)을 참조하세요.