---
displayed_sidebar: default
---

# 감사 로그
W&B 서버 감사 로그를 사용하여 팀 내의 사용자 활동을 추적하고 엔터프라이즈 거버넌스 요구 사항을 준수하세요. 감사 로그는 JSON 형식이며, 접근 방식은 W&B 서버 배포 유형에 따라 다릅니다:

| W&B 서버 배포 유형 | 감사 로그 접근 방식 |
|----------------------------|--------------------------------|
| 자체 관리 | 인스턴스 수준 버킷으로 10분마다 동기화됩니다. 또한 [API](#fetch-audit-logs-using-api)를 사용하여 접근할 수 있습니다. |
| 데디케이티드 클라우드 [보안 스토리지 커넥터(BYOB) 사용](./secure-storage-connector.md) | 인스턴스 수준 버킷(BYOB)으로 10분마다 동기화됩니다. 또한 [API](#fetch-audit-logs-using-api)를 사용하여 접근할 수 있습니다. |
| 데디케이티드 클라우드 W&B 관리 스토리지 사용(BYOB 없음) | [API](#fetch-audit-logs-using-api)를 사용하여만 접근할 수 있습니다. |

감사 로그에 접근하면 [Pandas](https://pandas.pydata.org/docs/index.html), [Amazon Redshift](https://aws.amazon.com/redshift/), [Google BigQuery](https://cloud.google.com/bigquery), [Microsoft Fabric](https://www.microsoft.com/en-us/microsoft-fabric) 등 선호하는 도구를 사용하여 분석할 수 있습니다. 분석 전에 JSON 형식의 감사 로그를 도구에 관련된 형식으로 변환해야 할 수 있습니다. 특정 도구에 대한 감사 로그 변환 정보는 W&B 문서의 범위를 벗어납니다.

:::tip
**감사 로그 보존:** 조직의 컴플라이언스, 보안 또는 위험 관리 팀이 특정 기간 동안 감사 로그를 보유하도록 요구하는 경우, W&B는 인스턴스 수준 버킷에서 로그를 장기 보존 스토리지로 정기적으로 전송할 것을 권장합니다. API를 사용하여 감사 로그에 접근하는 경우, 마지막 스크립트 실행 시간 이후 생성된 로그를 주기적으로(예: 매일 또는 며칠마다) 가져오고, 이러한 로그를 분석을 위한 단기 스토리지에 저장하거나 직접 장기 보존 스토리지로 전송하는 간단한 스크립트를 구현할 수 있습니다.
:::

:::note
**W&B 멀티-테넌트 SaaS 클라우드에 대한 감사 로그는 아직 사용할 수 없습니다.**
:::

## 감사 로그 스키마
다음 표는 감사 로그에 존재할 수 있는 다양한 키를 나열합니다. 각 로그는 해당 작업과 관련된 자산만 포함하며, 다른 자산은 로그에서 생략됩니다.

| 키 | 정의 |
|---------| -------|
|timestamp               | [RFC3339 형식](https://www.rfc-editor.org/rfc/rfc3339)의 타임스탬프. 예: `2023-01-23T12:34:56Z`, `2023년 1월 23일 UTC 시간 12:34:56`을 나타냅니다.
|action                  | 사용자가 수행한 [작업](#actions).
|actor_user_id           | 존재하는 경우, 작업을 수행한 로그인 된 사용자의 ID.
|response_code           | 작업에 대한 Http 응답 코드.
|artifact_asset          | 존재하는 경우, 이 아티팩트 ID에 대한 작업이 수행됨.
|artifact_sequence_asset | 존재하는 경우, 이 아티팩트 시퀀스 ID에 대한 작업이 수행됨.
|entity_asset            | 존재하는 경우, 이 엔터티 또는 팀 ID에 대한 작업이 수행됨.
|project_asset           | 존재하는 경우, 이 프로젝트 ID에 대한 작업이 수행됨.
|report_asset            | 존재하는 경우, 이 리포트 ID에 대한 작업이 수행됨.
|user_asset              | 존재하는 경우, 이 사용자 자산에 대한 작업이 수행됨.
|cli_version             | 작업이 python SDK를 통해 수행된 경우, 이 항목은 버전을 포함합니다.
|actor_ip                | 로그인 된 사용자의 IP 주소.
|actor_email             | 존재하는 경우, 이 액터 이메일에 대한 작업이 수행됨.
|artifact_digest         | 존재하는 경우, 이 아티팩트 다이제스트에 대한 작업이 수행됨.
|artifact_qualified_name | 존재하는 경우, 이 아티팩트에 대한 작업이 수행됨.
|entity_name             | 존재하는 경우, 이 엔터티 또는 팀 이름에 대한 작업이 수행됨.
|project_name            | 존재하는 경우, 이 프로젝트 이름에 대한 작업이 수행됨.
|report_name             | 존재하는 경우, 이 리포트 이름에 대한 작업이 수행됨.
|user_email              | 존재하는 경우, 이 사용자 이메일에 대한 작업이 수행됨.

개인 식별 정보(PII)와 같은 이메일 ID, 프로젝트, 팀 및 리포트 이름은 API 엔드포인트 옵션을 사용하여만 사용할 수 있으며, [아래에서 설명한](#fetch-audit-logs-using-api)대로 끌 수 있습니다.

## API를 사용한 감사 로그 가져오기
인스턴스 관리자는 다음 API를 사용하여 W&B 서버 인스턴스에 대한 감사 로그를 가져올 수 있습니다:
1. 기본 엔드포인트 `<wandb-server-url>/admin/audit_logs`와 다음 URL 파라미터의 조합을 사용하여 전체 API 엔드포인트를 구성합니다:
    - `numDays` : 로그는 `오늘 - numdays`부터 가장 최근까지 가져옵니다; 기본값은 `0` 즉, 오늘에 대한 로그만 반환됩니다.
    - `anonymize` : `true`로 설정하면, 모든 PII를 제거합니다; 기본값은 `false`입니다.
2. 구성된 전체 API 엔드포인트에 HTTP GET 요청을 실행합니다. 이는 현대적인 브라우저 내에서 직접 실행하거나, [Postman](https://www.postman.com/downloads/), [HTTPie](https://httpie.io/), cURL 명령 등의 도구를 사용하여 실행할 수 있습니다.

예를 들어, W&B 서버 인스턴스 URL이 `https://mycompany.wandb.io`이고 지난 주 동안의 사용자 활동에 대한 PII 없는 감사 로그를 얻고자 한다면, API 엔드포인트는 `https://mycompany.wandb.io?numDays=7&anonymize=true`가 됩니다.

:::note
API를 사용하여 감사 로그를 가져올 수 있는 것은 W&B 서버 [인스턴스 관리자](./manage-users.md#instance-admins)만 가능합니다. 인스턴스 관리자가 아니거나 조직에 로그인하지 않은 경우, `HTTP 403 Forbidden` 오류가 발생합니다.
:::

API 응답에는 줄 바꿈으로 구분된 JSON 객체가 포함됩니다. 객체에는 스키마에 설명된 필드가 포함됩니다. 이 형식은 인스턴스 수준 버킷으로 감사 로그 파일을 동기화할 때(앞서 언급한 대로 적용 가능한 경우) 사용되는 것과 동일합니다. 이러한 경우, 감사 로그는 버킷의 `/wandb-audit-logs` 디렉터리에 위치합니다.

## 작업
다음 표는 W&B에서 기록할 수 있는 가능한 작업을 설명합니다:

|작업 | 정의 |
|-----|-----|
| artifact:create             | 아티팩트가 생성됩니다.
| artifact:delete             | 아티팩트가 삭제됩니다.
| artifact:read               | 아티팩트가 읽혀집니다.
| project:delete              | 프로젝트가 삭제됩니다.
| project:read                | 프로젝트가 읽혀집니다.
| report:read                 | 리포트가 읽혀집니다.
| run:delete                  | 실행이 삭제됩니다.
| run:delete_many             | 실행들이 일괄적으로 삭제됩니다.
| run:update_many             | 실행들이 일괄적으로 업데이트됩니다.
| run:stop                    | 실행이 중단됩니다.
| run:undelete_many           | 실행들이 일괄적으로 휴지통에서 복구됩니다.
| run:update                  | 실행이 업데이트됩니다.
| sweep:create_agent          | 스윕 에이전트가 생성됩니다.
| team:invite_user            | 사용자가 팀에 초대됩니다.
| team:create_service_account | 팀을 위한 서비스 계정이 생성됩니다.
| team:create                 | 팀이 생성됩니다.
| team:uninvite               | 사용자 또는 서비스 계정이 팀에서 초대 취소됩니다.
| team:delete                 | 팀이 삭제됩니다.
| user:create                 | 사용자가 생성됩니다.
| user:delete_api_key         | 사용자의 API 키가 삭제됩니다.
| user:deactivate             | 사용자가 비활성화됩니다.
| user:create_api_key         | 사용자의 API 키가 생성됩니다.
| user:permanently_delete     | 사용자가 영구적으로 삭제됩니다.
| user:reactivate             | 사용자가 재활성화됩니다.
| user:update                 | 사용자가 업데이트됩니다.
| user:read                   | 사용자 프로필이 읽혀집니다.
| user:login                  | 사용자가 로그인합니다.
| user:initiate_login         | 사용자가 로그인을 시작합니다.
| user:logout                 | 사용자가 로그아웃합니다.