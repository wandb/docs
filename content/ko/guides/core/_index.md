---
title: W&B 코어
menu:
  default:
    identifier: ko-guides-core-_index
no_list: true
weight: 5
---

W&B Core는 [W&B Models]({{< relref path="/guides/models/" lang="ko" >}})와 [W&B Weave]({{< relref path="/guides/weave/" lang="ko" >}})를 지원하는 기본 프레임워크이며, 자체적으로는 [W&B Platform]({{< relref path="/guides/hosting/" lang="ko" >}})의 지원을 받습니다.

{{< img src="/images/general/core.png" alt="W&B Core 프레임워크 다이어그램" >}}

W&B Core는 전체 머신러닝 라이프사이클에 걸쳐 다양한 기능을 제공합니다. W&B Core를 사용하면 다음과 같은 작업이 가능합니다.

- 전체 계보 추적 기능으로 손쉬운 감사 및 재현성이 가능한 [ML 파이프라인 버전 관리 및 운영]({{< relref path="/guides/core/artifacts/" lang="ko" >}})이 가능합니다.
- [인터랙티브하고 구성 가능한 시각화 기능]({{< relref path="/guides/models/tables/" lang="ko" >}})을 이용해 데이터와 메트릭을 탐색하고 평가할 수 있습니다.
- 전체 조직에 걸쳐 누구나 이해하기 쉬운, 직관적이고 시각적인 포맷의 실시간 리포트를 생성해 [통찰을 문서화하고 공유]({{< relref path="/guides/core/reports/" lang="ko" >}})할 수 있습니다.
- [사용자 맞춤형 데이터 쿼리 및 시각화]({{< relref path="/guides/models/app/features/panels/query-panels/" lang="ko" >}})를 손쉽게 만들 수 있습니다.
- [시크릿을 활용해 민감한 문자열 보호]({{< relref path="/guides/core/secrets.md" lang="ko" >}})가 가능합니다.
- [모델 CI/CD]({{< relref path="/guides/core/automations/" lang="ko" >}})를 위한 주요 워크플로우를 트리거하는 자동화 설정이 가능합니다.