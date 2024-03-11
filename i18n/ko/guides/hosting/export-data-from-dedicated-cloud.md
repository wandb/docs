---
description: Export data from Dedicated Cloud
displayed_sidebar: default
---

# 전용 클라우드에서 데이터 내보내기

전용 클라우드 인스턴스에서 관리하는 모든 데이터를 내보내고 싶다면, W&B SDK API를 사용하여 run, 메트릭, 아티팩트 등을 추출하고, 해당 스토리지와 관련된 API를 사용하여 다른 클라우드 또는 온프레미스 스토리지에 로그할 수 있습니다.

데이터 내보내기 유스 케이스는 [데이터 가져오기 및 내보내기](../track/public-api-guide#export-data)에서 확인하세요. 또 다른 유스 케이스는 전용 클라우드 사용에 대한 계약을 종료하려는 경우, W&B가 인스턴스를 종료하기 전에 관련 데이터를 내보내고 싶을 수 있습니다.

아래 표를 참고하여 데이터 내보내기 API 및 관련 문서를 확인하세요:

| 목적 | 문서 |
|---------|---------------|
| 프로젝트 메타데이터 내보내기 | [프로젝트 API](../../ref/python/public-api/api#projects) |
| 프로젝트 내 run 내보내기 | [Run API](../../ref/python/public-api/api#runs), [Run 데이터 내보내기](../track/public-api-guide#export-run-data), [다수의 run 조회하기](../track/public-api-guide#querying-multiple-runs) |
| 리포트 내보내기 | [리포트 API](../../ref/python/public-api/api#reports) |
| 아티팩트 내보내기 | [아티팩트 API](../../ref/python/public-api/api#artifact), [아티팩트 그래프 탐색 및 트래버스하기](../artifacts/explore-and-traverse-an-artifact-graph#traverse-an-artifact-programmatically), [아티팩트 다운로드 및 사용하기](../artifacts/download-and-use-an-artifact#download-and-use-an-artifact-stored-on-wb) |

:::info
전용 클라우드에 저장된 아티팩트는 [보안 스토리지 커넥터](./secure-storage-connector)를 통해 관리합니다. 이 경우, W&B SDK API를 사용하여 아티팩트를 내보낼 필요가 없을 수 있습니다.
:::

:::note
W&B SDK API를 사용하여 모든 데이터를 내보내는 것은 run, 아티팩트 등의 수가 많을 경우 느릴 수 있습니다. 전용 클라우드 인스턴스를 과부하시키지 않도록 적절한 크기의 배치로 내보내기 프로세스를 실행하는 것이 W&B에서 권장합니다.
:::