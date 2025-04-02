---
title: Guides
description: W&B에 대한 개요와 처음 사용하는 사용자를 위한 시작 방법에 대한 링크입니다.
cascade:
  type: docs
menu:
  default:
    identifier: ko-guides-_index
    weight: 1
no_list: true
type: docs
---

## W&B 란 무엇인가요?

Weights & Biases (W&B) 는 AI 개발자 플랫폼으로, 모델 트레이닝, 모델 미세 조정 및 파운데이션 모델 활용을 위한 툴을 제공합니다.

{{< img src="/images/general/architecture.png" alt="" >}}

W&B 는 다음 세 가지 주요 구성 요소로 이루어져 있습니다: [Models]({{< relref path="/guides/models.md" lang="ko" >}}), [Weave](https://wandb.github.io/weave/), 그리고 [Core]({{< relref path="/guides/core/" lang="ko" >}}):

**[W&B Models]({{< relref path="/guides/models/" lang="ko" >}})** 는 머신러닝 개발자가 모델을 트레이닝하고 미세 조정하는 데 사용하는 가볍고 상호 운용 가능한 툴 모음입니다.
- [Experiments]({{< relref path="/guides/models/track/" lang="ko" >}}): 머신러닝 실험 추적
- [Sweeps]({{< relref path="/guides/models/sweeps/" lang="ko" >}}): 하이퍼파라미터 튜닝 및 모델 최적화
- [Registry]({{< relref path="/guides/core/registry/" lang="ko" >}}): ML 모델 및 데이터셋 게시 및 공유

**[W&B Weave]({{< relref path="/guides/weave/" lang="ko" >}})** 는 LLM 애플리케이션을 추적하고 평가하기 위한 가벼운 툴킷입니다.

**[W&B Core]({{< relref path="/guides/core/" lang="ko" >}})** 는 데이터 및 모델을 추적하고 시각화하며 결과를 전달하기 위한 강력한 빌딩 블록 세트입니다.
- [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ko" >}}): 에셋 버전 관리 및 계보 추적
- [Tables]({{< relref path="/guides/models/tables/" lang="ko" >}}): 테이블 형식 데이터 시각화 및 쿼리
- [Reports]({{< relref path="/guides/core/reports/" lang="ko" >}}): 발견한 내용 문서화 및 협업

## W&B 는 어떻게 작동하나요?

W&B 를 처음 사용하는 사용자이고 머신러닝 모델 및 실험을 트레이닝, 추적 및 시각화하는 데 관심이 있다면 다음 섹션을 순서대로 읽어보세요.

1. W&B 의 기본 연산 단위인 [runs]({{< relref path="/guides/models/track/runs/" lang="ko" >}})에 대해 알아보세요.
2. [Experiments]({{< relref path="/guides/models/track/" lang="ko" >}})로 머신러닝 실험을 생성하고 추적하세요.
3. [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ko" >}})로 데이터셋 및 모델 버전 관리를 위한 W&B 의 유연하고 가벼운 빌딩 블록을 찾아보세요.
4. [Sweeps]({{< relref path="/guides/models/sweeps/" lang="ko" >}})로 하이퍼파라미터 검색을 자동화하고 가능한 모델 공간을 탐색하세요.
5. [Registry]({{< relref path="/guides/core/registry/" lang="ko" >}})로 트레이닝에서 프로덕션까지 모델 수명 주기를 관리하세요.
6. [Data Visualization]({{< relref path="/guides/models/tables/" lang="ko" >}}) 가이드로 모델 버전 간 예측값을 시각화하세요.
7. [Reports]({{< relref path="/guides/core/reports/" lang="ko" >}})로 runs를 구성하고, 시각화를 포함 및 자동화하고, 발견한 내용을 설명하고, 협업자와 업데이트를 공유하세요.

<iframe width="100%" height="330" src="https://www.youtube.com/embed/tHAFujRhZLA" title="Weights &amp; Biases End-to-End Demo" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## W&B 를 처음 사용하시나요?

[quickstart]({{< relref path="/guides/quickstart/" lang="ko" >}})를 통해 W&B 를 설치하고 W&B 를 코드에 추가하는 방법을 알아보세요.
