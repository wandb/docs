---
title: 전용 클라우드에서 데이터 내보내기
description: 전용 클라우드에서 데이터 내보내기
menu:
  default:
    identifier: ko-guides-hosting-hosting-options-dedicated_cloud-export-data-from-dedicated-cloud
    parent: dedicated-cloud
url: guides/hosting/export-data-from-dedicated-cloud
---

전용 클라우드 인스턴스에 저장된 모든 데이터를 내보내고 싶다면, W&B SDK API를 사용하여 runs, metrics, artifacts 등 다양한 데이터를 [Import and Export API]({{< relref path="/ref/python/public-api/index.md" lang="ko" >}})로 추출할 수 있습니다. 아래 표는 주요 데이터 내보내기 유스 케이스와 관련된 문서를 정리한 것입니다.

| 목적 | 문서 |
|---------|---------------|
| 프로젝트 메타데이터 내보내기 | [Projects API]({{< relref path="/ref/python/public-api/projects.md" lang="ko" >}}) |
| 프로젝트 내 runs 내보내기 | [Runs API]({{< relref path="/ref/python/public-api/runs.md" lang="ko" >}}) |
| Reports 내보내기 | [Report and Workspace API]({{< relref path="/guides/core/reports/clone-and-export-reports/" lang="ko" >}}) |
| Artifacts 내보내기 | [Explore artifact graphs]({{< relref path="/guides/core/artifacts/explore-and-traverse-an-artifact-graph" lang="ko" >}}), [Download and use artifacts]({{< relref path="/guides/core/artifacts/download-and-use-an-artifact/#download-and-use-an-artifact-stored-on-wb" lang="ko" >}}) |

[Secure Storage Connector]({{< relref path="/guides/models/app/settings-page/teams/#secure-storage-connector" lang="ko" >}})를 활용하여 전용 클라우드에 저장된 artifacts를 관리하는 경우, 반드시 W&B SDK API를 이용해서 artifacts를 내보낼 필요가 없을 수도 있습니다.

{{% alert %}}
W&B SDK API를 사용해 모든 데이터를 내보낼 경우, runs, artifacts 등이 너무 많으면 속도가 느려질 수 있습니다. W&B는 전용 클라우드 인스턴스에 부하를 주지 않도록 내보내기 프로세스를 적정 크기의 배치로 나누어 실행할 것을 권장합니다.
{{% /alert %}}