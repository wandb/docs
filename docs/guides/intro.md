---
title: What is W&B?
description: W&B가 무엇인지 개요와 첫 번째 사용자라면 [시작 방법](./guides/get-started)에 대한 링크입니다.
slug: /guides
displayed_sidebar: default
---

Weights & Biases (W&B)는 AI 개발자 플랫폼으로, 모델 트레이닝, 모델 파인튜닝 및 기초 모델 활용을 위한 툴을 제공합니다.

![](/images/general/architecture.png)

W&B는 세 가지 주요 구성 요소로 이루어져 있습니다: [Models](/guides/models.md), [Weave](https://wandb.github.io/weave/), 그리고 [Core](/guides/core.md):

**[W&B Models](/guides/models.md)**는 기계학습 개발자가 모델을 트레이닝하고 파인튜닝하기 위한 경량의 상호운용 가능한 툴 세트입니다.
- [Experiments](/guides/track/intro.md): 기계학습 실험 추적
- [Sweeps](/guides/sweeps/intro.md): 하이퍼파라미터 튜닝 및 모델 최적화
- [Registry](/guides/registry/intro.md): ML 모델 및 데이터셋을 게시하고 공유
- [Launch](/guides/launch/intro.md): 작업의 확장 및 자동화

**[W&B Weave](https://wandb.github.io/weave/)**는 LLM 애플리케이션을 추적하고 평가하기 위한 경량 툴킷입니다.

**[W&B Core](/guides/core.md)**는 데이터와 모델을 추적하고 시각화하며, 결과를 전달하기 위한 강력한 빌딩 블록 세트입니다.
- [Artifacts](/guides/artifacts/intro.md): 자산의 버전 관리 및 계보 추적
- [Tables](/guides/tables/intro.md): 테이블형 데이터를 시각화하고 질의
- [Reports](/guides/reports/intro.md): 발견한 내용을 문서화하고 협업

## W&B는 어떻게 작동하나요?
W&B를 처음 사용하는 사용자이고, 기계학습 모델 및 실험의 트레이닝, 추적, 시각화에 관심이 있다면 다음 섹션을 순서대로 읽어보세요:

1. W&B의 기본 계산 단위인 [runs](./runs/intro.md)에 대해 알아보세요.
2. [Experiments](./track/intro.md)로 기계학습 실험을 생성하고 추적하세요.
3. 데이터셋 및 모델 버전 관리를 위한 W&B의 유연하고 경량의 빌딩 블록인 [Artifacts](./artifacts/intro.md)를 발견하세요.
4. [Sweeps](./sweeps/intro.md)를 통해 하이퍼파라미터 검색을 자동화하고 가능한 모델 공간을 탐색하세요.
5. 트레이닝부터 프로덕션까지의 모델 라이프사이클을 [Model Registry](./model_registry/intro.md)로 관리하세요.
6. [Data Visualization](./tables/intro.md) 가이드를 통해 모델 버전 간의 예측값을 시각화하세요.
7. [Reports](./reports/intro.md)로 runs를 조직하고, 시각화를 임베드 및 자동화하며, 발견 내용을 설명하고 협력자와 업데이트를 공유하세요.

<iframe width="100%" height="330" src="https://www.youtube.com/embed/tHAFujRhZLA" title="Weights &amp; Biases 엔드투엔드 데모" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## W&B를 처음 사용하시나요?

[퀵스타트](../quickstart.md)를 통해 W&B 설치 방법과 코드에 W&B를 추가하는 방법을 배워보세요.