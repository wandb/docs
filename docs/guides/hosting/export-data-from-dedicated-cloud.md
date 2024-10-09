---
title: Export data from Dedicated Cloud
description: 전용 클라우드에서 데이터 내보내기
displayed_sidebar: default
---

당신의 전용 클라우드 인스턴스에서 관리되는 모든 데이터를 내보내려면, W&B SDK API를 사용하여 runs, metrics, Artifacts 등을 추출하고 이를 해당 스토리지와 관련된 API를 사용하여 다른 클라우드 또는 온프레미스 스토리지에 로그할 수 있습니다.

데이터 내보내기 유스 케이스에 대해서는 [Import and Export Data](../track/public-api-guide#export-data)를 참조하세요. 또 다른 유스 케이스로는 전용 클라우드 사용 계약을 종료할 계획이라면 W&B가 인스턴스를 종료하기 전에 관련 데이터를 내보내고 싶을 수 있습니다. 

데이터 내보내기 API 및 관련 문서에 대한 포인터는 아래 표를 참조하세요:

| 목적 | 문서 |
|---------|---------------|
| 프로젝트 메타데이터 내보내기 | [Projects API](../../ref/python/public-api/api#projects) |
| 프로젝트 내에서 runs 내보내기 | [Runs API](../../ref/python/public-api/api#runs), [Export run data](../track/public-api-guide#export-run-data), [Querying multiple runs](../track/public-api-guide#querying-multiple-runs) |
| Reports 내보내기 | [Reports API](../../ref/python/public-api/api#reports) |
| Artifacts 내보내기 | [Artifact API](../../ref/python/public-api/api#artifact), [Explore and traverse an artifact graph](../artifacts/explore-and-traverse-an-artifact-graph/#use-the-api-to-track-lineage), [Download and use an artifact](../artifacts/download-and-use-an-artifact#download-and-use-an-artifact-stored-on-wb) |

:::info
전용 클라우드에 저장된 Artifacts는 [Secure Storage Connector](./data-security/secure-storage-connector)로 관리합니다. 이 경우, W&B SDK API를 사용하여 Artifacts를 내보낼 필요가 없을 수도 있습니다.
:::

:::note
W&B SDK API를 사용하여 모든 데이터를 내보내는 것은 많은 수의 runs, Artifacts 등이 있는 경우 느릴 수 있습니다. W&B는 적절히 크기가 조정된 배치로 내보내기 프로세스를 실행하여 전용 클라우드 인스턴스의 부담을 덜어줄 것을 권장합니다.
:::
