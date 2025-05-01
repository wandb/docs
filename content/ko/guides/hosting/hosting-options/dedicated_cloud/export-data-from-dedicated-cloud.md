---
title: Export data from Dedicated cloud
description: 전용 클라우드에서 데이터 내보내기
menu:
  default:
    identifier: ko-guides-hosting-hosting-options-dedicated_cloud-export-data-from-dedicated-cloud
    parent: dedicated-cloud
url: /ko/guides//hosting/export-data-from-dedicated-cloud
---

전용 클라우드 인스턴스에서 관리하는 모든 데이터를 내보내려면 W&B SDK API를 사용하여 [Import and Export API]({{< relref path="/ref/python/public-api/" lang="ko" >}})로 run, 메트릭, Artifacts 등을 추출할 수 있습니다. 다음 표는 주요 내보내기 유스 케이스를 다룹니다.

| 목적 | 문서 |
|---------|---------------|
| 프로젝트 메타데이터 내보내기 | [Projects API]({{< relref path="/ref/python/public-api/projects/" lang="ko" >}}) |
| 프로젝트에서 run 내보내기 | [Runs API]({{< relref path="/ref/python/public-api/runs/" lang="ko" >}}) |
| Reports 내보내기 | [Reports API]({{< relref path="/guides/core/reports/clone-and-export-reports/" lang="ko" >}}) |
| Artifacts 내보내기 | [Artifact 그래프 탐색]({{< relref path="/guides/core/artifacts/explore-and-traverse-an-artifact-graph" lang="ko" >}}), [Artifacts 다운로드 및 사용]({{< relref path="/guides/core/artifacts/download-and-use-an-artifact/#download-and-use-an-artifact-stored-on-wb" lang="ko" >}}) |

[Secure Storage Connector]({{< relref path="/guides/models/app/settings-page/teams/#secure-storage-connector" lang="ko" >}})를 사용하여 전용 클라우드에 저장된 Artifacts를 관리하는 경우 W&B SDK API를 사용하여 Artifacts를 내보낼 필요가 없을 수 있습니다.

{{% alert %}}
W&B SDK API를 사용하여 모든 데이터를 내보내는 것은 run, Artifacts 등이 많은 경우 속도가 느릴 수 있습니다. W&B는 전용 클라우드 인스턴스에 과부하가 걸리지 않도록 적절한 크기의 배치로 내보내기 프로세스를 실행하는 것이 좋습니다.
{{% /alert %}}
