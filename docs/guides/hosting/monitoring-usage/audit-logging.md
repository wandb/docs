---
title: Track user activity with audit logs
displayed_sidebar: default
---

W&B Server 감사 로그를 사용하여 팀 내 사용자 활동을 추적하고, 기업 거버넌스 요구 사항을 준수하십시오. 감사 로그는 JSON 형식이며, 액세스 메커니즘은 W&B Server 배포 유형에 따라 다릅니다:

| W&B Server 배포 유형 | 감사 로그 엑세스 메커니즘 |
|----------------------|----------------------------|
| 자체 관리형 | 10분마다 인스턴스 수준 버킷으로 동기화됩니다. [API](#fetch-audit-logs-using-api)를 사용하여 액세스할 수도 있습니다. |
| [보안 스토리지 커넥터 (BYOB)](../data-security/secure-storage-connector.md)가 있는 전용 클라우드 | 10분마다 인스턴스 수준 버킷 (BYOB)으로 동기화됩니다. [API](#fetch-audit-logs-using-api)를 사용하여 액세스할 수도 있습니다. |
| W&B 관리 스토리지 (BYOB 없이)가 있는 전용 클라우드 | [API](#fetch-audit-logs-using-api)를 사용하여서만 액세스할 수 있습니다. |

감사 로그에 엑세스한 후, [Pandas](https://pandas.pydata.org/docs/index.html), [Amazon Redshift](https://aws.amazon.com/redshift/), [Google BigQuery](https://cloud.google.com/bigquery), [Microsoft Fabric](https://www.microsoft.com/en-us/microsoft-fabric) 등 선호하는 툴을 사용하여 로그를 분석하십시오. 분석 전에 JSON 형식의 감사 로그를 툴에 적합한 형식으로 변환해야 할 수도 있습니다. 특정 툴에 대한 감사 로그 변환 방법에 대한 정보는 W&B 문서 범위 밖입니다.

:::tip
**감사 로그 보존:** 조직 내 준수, 보안 또는 위험 팀에서 특정 기간 동안 감사 로그를 보관해야 하는 경우, W&B는 인스턴스 수준 버킷에서 장기 보존 스토리지로 주기적으로 로그를 전송할 것을 권장합니다. API를 사용하여 감사 로그를 엑세스하는 경우, 간단한 스크립트를 구현하여 주기적으로 (예: 매일 또는 며칠마다) 마지막 스크립트 실행 이후 생성된 로그를 가져와 분석을 위해 단기 보관 스토리지에 저장하거나 장기 보존 스토리지로 직접 전송할 수 있습니다.
:::

:::note
감사 로그는 아직 W&B 멀티 테넌트 클라우드에서는 사용할 수 없습니다.
:::

## 감사 로그 스키마
다음 표는 감사 로그에 있을 수 있는 모든 키를 나열합니다. 각 로그에는 해당 작업과 관련된 자산만 포함되며, 다른 자산은 로그에서 생략됩니다.

| 키 | 정의 |
|----|-----|
|timestamp               | [RFC3339 형식](https://www.rfc-editor.org/rfc/rfc3339)의 타임스탬프입니다. 예: `2023-01-23T12:34:56Z`, 이는 `2023년 1월 23일`의 `12:34:56 UTC` 시간을 나타냅니다.
|action                  | 사용자가 수행한 [작업](#actions)입니다.
|actor_user_id           | 존재할 경우, 작업을 수행한 로그인한 사용자의 ID입니다.
|response_code           | 작업에 대한 Http 응답 코드입니다.
|artifact_asset          | 존재할 경우, 이 아티팩트 id에서 작업이 수행되었습니다.
|artifact_sequence_asset | 존재할 경우, 이 아티팩트 시퀀스 id에서 작업이 수행되었습니다.
|entity_asset            | 존재할 경우, 이 엔티티 또는 팀 id에서 작업이 수행되었습니다.
|project_asset           | 존재할 경우, 이 프로젝트 id에서 작업이 수행되었습니다.
|report_asset            | 존재할 경우, 이 리포트 id에서 작업이 수행되었습니다.
|user_asset              | 존재할 경우, 이 사용자 자산에서 작업이 수행되었습니다.
|cli_version             | 작업이 python SDK를 통해 수행된 경우, 이 버전을 포함합니다.
|actor_ip                | 로그인한 사용자의 IP 어드레스입니다.
|actor_email             | 존재할 경우, 이 배우 이메일에서 작업이 수행되었습니다.
|artifact_digest         | 존재할 경우, 이 아티팩트 다이제스트에서 작업이 수행되었습니다.
|artifact_qualified_name | 존재할 경우, 이 아티팩트에서 작업이 수행되었습니다.
|entity_name             | 존재할 경우, 이 엔티티 또는 팀 이름에서 작업이 수행되었습니다.
|project_name            | 존재할 경우, 이 프로젝트 이름에서 작업이 수행되었습니다.
|report_name             | 존재할 경우, 이 리포트 이름에서 작업이 수행되었습니다.
|user_email              | 존재할 경우, 이 사용자 이메일에서 작업이 수행되었습니다.

개인 식별 정보(PII)와 같은 이메일 ID, 프로젝트, 팀 및 리포트 이름은 API 엔드포인트 옵션을 사용해야만 사용 가능하며, [아래에 설명된 대로](#fetch-audit-logs-using-api) 비활성화할 수 있습니다.

## API를 사용한 감사 로그 가져오기
인스턴스 관리자는 다음 API를 사용하여 W&B 서버 인스턴스의 감사 로그를 가져올 수 있습니다:
1. 기본 엔드포인트 `<wandb-server-url>/admin/audit_logs`와 다음 URL 파라미터를 조합하여 전체 API 엔드포인트를 구성하십시오:
    - `numDays` : `오늘 - numdays`부터 최신까지의 로그가 가져옵니다; 기본값은 `0`으로, 즉 `오늘`의 로그만 반환됩니다.
    - `anonymize` : `true`로 설정된 경우, PII를 제거합니다; 기본값은 `false`입니다.
2. 구성된 전체 API 엔드포인트에서 HTTP GET 요청을 실행합니다. 이는 최신 브라우저 내에서 직접 실행하거나 [Postman](https://www.postman.com/downloads/), [HTTPie](https://httpie.io/), cURL 코맨드 등과 같은 툴을 사용하여 실행할 수 있습니다.

W&B Server 인스턴스 URL이 `https://mycompany.wandb.io`이고 최근 1주일 동안의 사용자 활동에 대한 PII 없는 감사 로그를 가져오고 싶다면, API 엔드포인트는 `https://mycompany.wandb.io?numDays=7&anonymize=true`가 됩니다.

:::note
오직 W&B Server [인스턴스 관리자](../iam/manage-users.md#instance-admins)만 API를 사용하여 감사 로그를 가져올 수 있습니다. 인스턴스 관리자가 아니거나 조직에 로그인하지 않은 경우, `HTTP 403 Forbidden` 오류가 발생합니다.
:::

API 응답은 줄바꿈으로 구분된 JSON 오브젝트를 포함합니다. 오브젝트에는 스키마에 설명된 필드가 포함됩니다. 이는 인스턴스 수준 버킷에 감사 로그 파일을 동기화할 때 사용된 것과 동일한 형식입니다 (앞서 설명한 대로 해당하는 경우). 그러한 경우, 감사 로그는 `/wandb-audit-logs` 디렉토리 내의 버킷에 위치합니다.

## Actions
다음 표는 W&B에 의해 기록될 수 있는 가능한 작업을 설명합니다:

|작업 | 정의 |
|-----|-----|
| artifact:create             | 아티팩트가 생성됩니다.
| artifact:delete             | 아티팩트가 삭제됩니다.
| artifact:read               | 아티팩트가 읽힙니다.
| project:delete              | 프로젝트가 삭제됩니다.
| project:read                | 프로젝트가 읽힙니다.
| report:read                 | 리포트가 읽힙니다.
| run:delete                  | Run이 삭제됩니다.
| run:delete_many             | 여러 Runs가 일괄 삭제됩니다.
| run:update_many             | 여러 Runs가 일괄 업데이트됩니다.
| run:stop                    | Run이 중지됩니다.
| run:undelete_many           | 여러 Runs가 휴지통에서 일괄 복원됩니다.
| run:update                  | Run이 업데이트됩니다.
| sweep:create_agent          | Sweep 에이전트가 생성됩니다.
| team:invite_user            | 사용자가 팀에 초대됩니다.
| team:create_service_account | 팀의 서비스 계정이 생성됩니다.
| team:create                 | 팀이 생성됩니다.
| team:uninvite               | 사용자 또는 서비스 계정이 팀에서 초대 취소됩니다.
| team:delete                 | 팀이 삭제됩니다.
| user:create                 | 사용자가 생성됩니다.
| user:delete_api_key         | 사용자에 대한 API 키가 삭제됩니다.
| user:deactivate             | 사용자가 비활성화됩니다.
| user:create_api_key         | 사용자에 대한 API 키가 생성됩니다.
| user:permanently_delete     | 사용자가 영구적으로 삭제됩니다.
| user:reactivate             | 사용자가 재활성화됩니다.
| user:update                 | 사용자가 업데이트됩니다.
| user:read                   | 사용자 프로필이 읽힙니다.
| user:login                  | 사용자가 로그인합니다.
| user:initiate_login         | 사용자가 로그인을 시작합니다.
| user:logout                 | 사용자가 로그아웃합니다.