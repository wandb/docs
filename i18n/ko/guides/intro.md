---
description: An overview of what is W&B along with links on how to get started if
  you are a first time user.
slug: /guides
displayed_sidebar: default
---

# W&B란 무엇인가요?

Weights & Biases(W&B)는 모델 트레이닝, 파인 튜닝 및 파운데이션 모델 활용을 위한 AI 개발자 플랫폼입니다.

W&B를 5분 만에 설치한 후, 신뢰할 수 있는 기록 시스템에 모델과 데이터가 추적되고 버전 관리되고 있다는 확신을 가지고 머신러닝 파이프라인을 빠르게 반복하세요.

![](@site/static/images/general/architecture.png)

이 다이어그램은 W&B 제품 간의 관계를 간략하게 설명합니다.

**[W&B Models](/guides/models.md)**는 모델 트레이닝 및 파인 튜닝을 위한 가볍고 상호 운용 가능한 도구 세트입니다.
- [Experiments](/guides/track/intro.md): 기계학습 실험 추적
- [Model Registry](/guides/model_registry/intro.md): 프로덕션 모델의 중앙 관리
- [Launch](/guides/launch/intro.md): 워크로드의 확장과 자동화
- [Sweeps](/guides/sweeps/intro.md): 하이퍼파라미터 튜닝 및 모델 최적화

**[W&B Prompts](/guides/prompts/intro.md)**는 LLMs의 디버깅 및 평가를 위한 것입니다.

**[W&B Platform](/guides/platform.md)**은 데이터와 모델을 추적하고 시각화하며 결과를 공유하는 기능을 위한 강력한 기본 구성 요소 세트입니다.
- [Artifacts](/guides/artifacts/intro.md): 자산 버전 관리 및 계보 추적
- [Tables](/guides/tables/intro.md): 표 형식 데이터의 시각화 및 쿼리
- [Reports](/guides/reports/intro.md): 발견한 내용의 문서화 및 협업
- [Weave](/guides/app/features/panels/weave) 데이터의 쿼리 및 시각화

## W&B를 처음 사용하시나요?

<iframe width="100%" height="330" src="https://www.youtube.com/embed/tHAFujRhZLA" title="Weights &amp; Biases End-to-End Demo" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

다음 자료를 통해 W&B를 살펴 보세요:

1. [Intro Notebook](http://wandb.me/intro): 간단한 샘플 코드를 실행하여 5분 안에 실험을 추적해 보세요.
2. [퀵스타트](../quickstart.md): 코드에 W&B를 어디에 어떻게 추가해야 하는지에 대한 간단한 개요를 읽어보세요.
1. [인테그레이션 가이드](./integrations/intro.md) 와 [W&B 간단한 인테그레이션 YouTube](https://www.youtube.com/playlist?list=PLD80i8An1OEGDADxOBaH71ZwieZ9nmPGC) 플레이리스트를 둘러보며 선호하는 머신러닝 프레임워크와 W&B를 통합하는 방법에 대한 정보를 얻으세요.
1. W&B Python Library, CLI 및 Weave 작업에 대한 기술 사양은 [API 참조 가이드](../ref/README.md)를 참조하세요.

## W&B는 어떻게 작동하나요?

W&B를 처음 사용하는 경우 이 섹션을 순서대로 읽는 것을 추천해 드립니다:

1. W&B의 기본 계산 단위인 [Runs](./runs/intro.md)에 대해 알아보세요.
2. [Experiments](./track/intro.md)를 사용하여 머신러닝 실험을 생성하고 추적하세요.
3. 데이터셋 및 모델 버전 관리를 위한 W&B의 유연하고 가벼운 빌딩 블록인 [Artifacts](./artifacts/intro.md)에 대해 알아보세요.
4. [Sweeps](./sweeps/intro.md)로 하이퍼파라미터 검색을 자동화하고 모델의 가능성을 탐색하세요.
5. [Model Management](./model_registry/intro.md)로 트레이닝부터 프로덕션까지 모델 라이프라이클을 관리하세요.
6. [Data Visualization](./tables/intro.md) 가이드를 통해 모델 버전별 예측값을 시각화하세요.
7. [Reports](./reports/intro.md)로 W&B Runs를 구성하고 시각화를 내장화 및 자동화하며, 발견한 내용을 설명하고 협업자와 업데이트를 공유하세요.