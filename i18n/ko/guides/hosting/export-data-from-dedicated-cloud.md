---
description: Export data from Dedicated Cloud
displayed_sidebar: default
---

# 데디케이티드 클라우드에서 데이터 내보내기

데디케이티드 클라우드 인스턴스에서 관리하는 모든 데이터를 내보고 싶다면, W&B SDK API를 사용하여 실행, 메트릭, 아티팩트 등을 추출하고, 해당 저장소와 관련된 API를 사용하여 다른 클라우드 또는 온-프레미스 저장소에 로그할 수 있습니다.

데이터 내보내기 사용 사례는 [데이터 가져오기 및 내보내기](../track/public-api-guide#export-data)에서 확인하세요. 다른 사용 사례는 데디케이티드 클라우드 사용에 대한 계약을 종료할 계획이라면, W&B가 인스턴스를 종료하기 전에 관련 데이터를 내보내고자 할 수 있습니다.

아래 표를 참조하여 데이터 내보내기 API 및 관련 문서를 확인하세요:

| 목적 | 문서 |
|---------|---------------|
| 프로젝트 메타데이터 내보내기 | [프로젝트 API](../../ref/python/public-api/api#projects) |
| 프로젝트의 실행 내보내기 | [실행 API](../../ref/python/public-api/api#runs), [실행 데이터 내보내기](../track/public-api-guide#export-run-data), [여러 실행 쿼리하기](../track/public-api-guide#querying-multiple-runs) |
| 리포트 내보내기 | [리포트 API](../../ref/python/public-api/api#reports) |
| 아티팩트 내보내기 | [아티팩트 API](../../ref/python/public-api/api#artifact), [아티팩트 그래프 탐색 및 트래버스](../artifacts/explore-and-traverse-an-artifact-graph#traverse-an-artifact-programmatically), [아티팩트 다운로드 및 사용](../artifacts/download-and-use-an-artifact#download-and-use-an-artifact-stored-on-wb) |

:::info
데디케이티드 클라우드에 저장된 아티팩트는 [보안 저장소 커넥터](./secure-storage-connector)로 관리합니다. 이 경우, W&B SDK API를 사용하여 아티팩트를 내보낼 필요가 없을 수 있습니다.
:::

:::note
실행, 아티팩트 등이 많은 경우 W&B SDK API를 사용하여 모든 데이터를 내보내는 것은 느릴 수 있습니다. 데디케이티드 클라우드 인스턴스를 압도하지 않도록 적절한 크기의 배치로 내보내기 프로세스를 실행하는 것이 W&B에서 권장합니다.
:::