---
title: W&B 앱 UI
aliases:
- /guides/models/app/features
menu:
  default:
    identifier: ko-guides-models-app-_index
    parent: models
url: guides/app
---

이 섹션에서는 W&B App UI 사용에 도움이 되는 세부 정보를 안내합니다. Workspace, Teams, Registry 를 관리하고, Experiments 를 시각화 및 모니터링하며, 패널과 Reports 를 생성하고, 자동화를 구성하는 등 다양한 작업이 가능합니다.

웹 브라우저에서 W&B App 에 접속하세요.

- W&B 멀티 테넌트 배포는 https://wandb.ai/ 에서 퍼블릭 웹을 통해 접속할 수 있습니다.
- W&B Dedicated Cloud 배포는 Dedicated Cloud 가입 시 설정한 도메인에서 접속할 수 있습니다. 관리자 사용자는 W&B Management Console 에서 도메인을 업데이트할 수 있습니다. 오른쪽 상단 아이콘을 클릭한 뒤 **System console** 을 선택하세요.
- W&B Self-Managed 배포는 W&B 설치 시 지정한 호스트명에서 접속합니다. 예를 들어 Helm 으로 배포한 경우 호스트명은 `values.global.host` 에서 설정합니다. 관리자 사용자는 W&B Management Console 에서 도메인을 업데이트할 수 있습니다. 오른쪽 상단 아이콘을 클릭한 뒤 **System console** 을 선택하세요.

더 알아보기:

- [실험 추적하기]({{< relref path="/guides/models/track/" lang="ko" >}}): Runs 또는 Sweeps 를 사용해 실험을 추적하세요.
- [배포 설정 구성하기]({{< relref path="settings-page/" lang="ko" >}}) 및 [기본값 설정]({{< relref path="features/cascade-settings.md" lang="ko" >}}).
- [패널 추가하기]({{< relref path="/guides/models/app/features/panels/" lang="ko" >}}): 실험을 시각화하세요. 예: 선 그래프, 막대 그래프, 미디어 패널, 쿼리 패널, 테이블 등.
- [커스텀 차트 추가하기]({{< relref path="/guides/models/app/features/custom-charts/" lang="ko" >}}).
- [Reports 생성 및 공유하기]({{< relref path="/guides/core/reports/" lang="ko" >}}).