---
description: An overview of what is W&B along with links on how to get started if
  you are a first time user.
slug: /guides
displayed_sidebar: default
---

# W&B란 무엇인가요?

Weights & Biases (W&B)는 모델 학습, 모델 세부 조정 및 기반 모델 활용을 위한 도구를 제공하는 AI 개발자 플랫폼입니다.

W&B를 5분 안에 설정한 다음, 모델과 데이터가 신뢰할 수 있는 기록 시스템에 추적되고 버전 관리되는 것을 확신하며 머신 러닝 파이프라인을 빠르게 반복하세요.

![](@site/static/images/general/architecture.png)

이 다이어그램은 W&B 제품 간의 관계를 개략적으로 설명합니다.

**[W&B Models](/guides/models.md)**는 모델 학습 및 세부 조정을 위한 경량, 상호 운용 가능한 도구 세트입니다.
- [실험](/guides/track/intro.md): 머신 러닝 실험 추적
- [모델 레지스트리](/guides/model_registry/intro.md): 프로덕션 모델 중앙 관리
- [런치](/guides/launch/intro.md): 워크로드 규모 확장 및 자동화
- [스윕](/guides/sweeps/intro.md): 하이퍼파라미터 튜닝 및 모델 최적화

**[W&B Prompts](/guides/prompts/intro.md)**는 LLMs 디버깅 및 평가를 위한 것입니다.

**[W&B 플랫폼](/guides/platform.md)**은 데이터 및 모델 추적 및 시각화, 결과 공유를 위한 강력한 기본 구성 요소 세트입니다.
- [아티팩트](/guides/artifacts/intro.md): 자산 버전 관리 및 계보 추적
- [테이블](/guides/tables/intro.md): 테이블 데이터 시각화 및 쿼리
- [리포트](/guides/reports/intro.md): 발견한 내용 문서화 및 협업
- [위브](/guides/app/features/panels/weave) 데이터 쿼리 및 시각화 생성

## W&B를 처음 사용하시나요?

<iframe width="100%" height="330" src="https://www.youtube.com/embed/tHAFujRhZLA" title="Weights &amp; Biases End-to-End Demo" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

이러한 리소스로 W&B 탐색을 시작하세요:

1. [소개 노트북](http://wandb.me/intro): 5분 만에 실험 추적을 위한 간단한 샘플 코드 실행
2. [퀵스타트](../quickstart.md): W&B를 코드에 어떻게 추가할지에 대한 간략한 개요 읽기
1. [통합 가이드](./integrations/intro.md) 및 [W&B 쉬운 통합 YouTube](https://www.youtube.com/playlist?list=PLD80i8An1OEGDADxOBaH71ZwieZ9nmPGC) 재생목록을 탐색하여 선호하는 머신 러닝 프레임워크와 W&B를 통합하는 방법에 대한 정보를 얻으세요.
1. W&B Python 라이브러리, CLI 및 Weave 작업에 대한 기술 사양에 대한 [API 참조 가이드](../ref/README.md)를 확인하세요.

## W&B는 어떻게 작동하나요?

W&B를 처음 사용하는 경우 다음 섹션을 이 순서대로 읽는 것이 좋습니다:

1. W&B의 기본 계산 단위인 [실행](./runs/intro.md)에 대해 알아보세요.
2. [실험](./track/intro.md)을 사용하여 머신 러닝 실험을 생성 및 추적하세요.
3. 데이터세트 및 모델 버전 관리를 위한 W&B의 유연하고 경량의 구성 요소인 [아티팩트](./artifacts/intro.md)를 발견하세요.
4. [스윕](./sweeps/intro.md)으로 하이퍼파라미터 검색을 자동화하고 가능한 모델의 공간을 탐색하세요.
5. [모델 관리](./model_registry/intro.md)를 통해 학습부터 프로덕션까지 모델 수명주기를 관리하세요.
6. 모델 버전 간 예측값을 시각화하는 [데이터 시각화](./tables/intro.md) 가이드를 확인하세요.
7. [리포트](./reports/intro.md)로 W&B 실행을 정리하고, 시각화를 임베드 및 자동화하며, 발견한 내용을 설명하고, 협업자와 업데이트를 공유하세요.