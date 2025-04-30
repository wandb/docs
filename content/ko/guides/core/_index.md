---
title: W&B Core
menu:
  default:
    identifier: ko-guides-core-_index
no_list: true
weight: 5
---

W&B Core는 [W&B Models]({{< relref path="/guides/models/" lang="ko" >}}) 와 [W&B Weave]({{< relref path="/guides/weave/" lang="ko" >}}) 를 지원하는 기본 프레임워크이며, [W&B Platform]({{< relref path="/guides/hosting/" lang="ko" >}}) 에 의해 자체적으로 지원됩니다.

{{< img src="/images/general/core.png" alt="" >}}

W&B Core는 전체 ML 라이프사이클에 걸쳐 기능을 제공합니다. W&B Core를 통해 다음을 수행할 수 있습니다.

- 쉬운 감사 및 재현성을 위해 전체 계보 추적과 함께 ML [파이프라인을 버전 관리하고 관리]({{< relref path="/guides/core/artifacts/" lang="ko" >}}) 할 수 있습니다.
- [상호작용할 수 있는 시각화 기능]({{< relref path="/guides/models/tables/" lang="ko" >}})을 사용하여 데이터와 메트릭을 탐색하고 평가합니다.
- 비기술적 이해 관계자가 쉽게 이해할 수 있는 소화하기 쉽고 시각적인 형식으로 라이브 리포트 를 생성하여 조직 전체에서 [인사이트를 문서화하고 공유]({{< relref path="/guides/core/reports/" lang="ko" >}}) 할 수 있습니다.
- 사용자 정의 요구 사항을 충족하는 [데이터 시각화를 쿼리하고 생성]({{< relref path="/guides/models/app/features/panels/query-panels/" lang="ko" >}}) 합니다.
- [secret 을 사용하여 중요한 문자열을 보호]({{< relref path="/guides/core/secrets.md" lang="ko" >}}) 합니다.
- [모델 CI/CD]({{< relref path="/guides/core/automations/" lang="ko" >}}) 를 위한 주요 워크플로우 를 트리거하는 자동화를 구성합니다.
