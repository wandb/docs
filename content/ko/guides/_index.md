---
title: 가이드
description: W&B 개요와 시작 방법 안내
cascade:
  type: docs
menu:
  default:
    identifier: ko-guides-_index
    weight: 1
no_list: true
type: docs
---

## W&B란 무엇인가요?

W&B는 모델 트레이닝, 파인튜닝, 그리고 파운데이션 모델을 활용할 수 있는 AI 개발자 플랫폼입니다.

{{< img src="/images/general/architecture.png" alt="W&B 플랫폼 아키텍처 다이어그램" >}}

W&B는 세 가지 주요 구성요소로 이루어져 있습니다: [Models]({{< relref path="/guides/models.md" lang="ko" >}}), [Weave](https://wandb.github.io/weave/), 그리고 [Core]({{< relref path="/guides/core/" lang="ko" >}}):

**[W&B Models]({{< relref path="/guides/models/" lang="ko" >}})** 는 기계학습 개발자들이 모델을 트레이닝하고 파인튜닝할 때 사용하는 가볍고 상호운용성 높은 툴셋입니다.
- [Experiments]({{< relref path="/guides/models/track/" lang="ko" >}}): 기계학습 실험 추적
- [Sweeps]({{< relref path="/guides/models/sweeps/" lang="ko" >}}): 하이퍼파라미터 튜닝 및 모델 최적화
- [Registry]({{< relref path="/guides/core/registry/" lang="ko" >}}): ML 모델 및 데이터셋을 퍼블리시하고 공유

**[W&B Weave]({{< relref path="/guides/weave/" lang="ko" >}})** 는 LLM 애플리케이션을 추적하고 평가할 수 있는 경량화된 툴킷입니다.

**[W&B Core]({{< relref path="/guides/core/" lang="ko" >}})** 는 데이터와 모델을 추적·시각화하고, 결과를 효과적으로 공유할 수 있는 강력한 빌딩 블록 모음입니다.
- [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ko" >}}): 자산의 버전 관리 및 계보 추적
- [Tables]({{< relref path="/guides/models/tables/" lang="ko" >}}): 표 형태의 데이터 시각화 및 쿼리
- [Reports]({{< relref path="/guides/core/reports/" lang="ko" >}}): 발견한 내용을 문서화하고 협업

{{% alert %}}
[W&B 릴리즈 노트]({{< relref path="/ref/release-notes/" lang="ko" >}})에서 최근 릴리즈 소식을 확인하세요.
{{% /alert %}}

## W&B는 어떻게 동작하나요?

W&B를 처음 사용하고, 기계학습 모델과 실험을 트레이닝·추적·시각화하는 데 관심이 있다면 아래 섹션을 순서대로 읽어보세요.

1. [runs]({{< relref path="/guides/models/track/runs/" lang="ko" >}})에 대해 알아보기 — W&B의 기본 연산 단위입니다.
2. [Experiments]({{< relref path="/guides/models/track/" lang="ko" >}})로 기계학습 실험을 생성하고 추적하기
3. [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ko" >}})로 데이터셋과 모델의 유연하고 경량화된 버전 관리 방법 발견하기
4. [Sweeps]({{< relref path="/guides/models/sweeps/" lang="ko" >}})로 하이퍼파라미터 탐색을 자동화하고 다양한 모델을 탐색하기
5. [Registry]({{< relref path="/guides/core/registry/" lang="ko" >}})로 모델의 전체 라이프사이클을 트레이닝부터 프로덕션까지 관리하기
6. [Data Visualization]({{< relref path="/guides/models/tables/" lang="ko" >}}) 가이드로 다양한 모델 버전에서 예측값 시각화하기
7. [Reports]({{< relref path="/guides/core/reports/" lang="ko" >}})로 run을 정리하고, 시각화 결과를 자동화 및 임베딩하며, 발견한 내용을 설명하고 협업자와 업데이트를 공유하기

<iframe width="100%" height="330" src="https://www.youtube.com/embed/tHAFujRhZLA" title="Weights &amp; Biases End-to-End Demo" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## W&B가 처음이신가요?

[퀵스타트]({{< relref path="/guides/quickstart/" lang="ko" >}})에서 W&B 설치 방법과 W&B를 코드에 적용하는 방법을 확인해 보세요.