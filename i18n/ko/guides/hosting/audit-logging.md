---
displayed_sidebar: default
---

# 감사 로그
W&B 서버 감사 로그를 사용하여 팀 내 사용자 활동을 추적하고 엔터프라이즈 거버넌스 요구 사항을 충족하세요. 감사 로그는 JSON 형식이며, 엑세스 방식은 W&B 서버 배포 유형에 따라 다릅니다:

| W&B 서버 배포 유형 | 감사 로그 엑세스 방식 |
|----------------------------|--------------------------------|
| 자체 관리 | 인스턴스 수준 버킷에 10분마다 동기화됩니다. [API](#fetch-audit-logs-using-api)를 사용하여 엑세스할 수도 있습니다. |
| 전용 클라우드 [secure storage connector (BYOB)](./secure-storage-connector.md) | 인스턴스 수준 버킷(BYOB)에 10분마다 동기화됩니다. [API](#fetch-audit-logs-using-api)를 사용하여 엑세스할 수도 있습니다. |
| 전용 클라우드 W&B 관리 스토리지(BYOB 없음) | [API](#fetch-audit-logs-using-api)를 사용하여만 엑세스할 수 있습니다. |

감사 로그에 엑세스하면, [Pandas](https://pandas.pydata.org/docs/index.html), [Amazon Redshift](https://aws.amazon.com/redshift/), [Google BigQuery](https://cloud.google.com/bigquery), [Microsoft Fabric](https://www.microsoft.com/en-us/microsoft-fabric) 등과 같은 선호하는 툴을 사용하여 분석할 수 있습니다. 분석을 위해 JSON 형식의 감사 로그를 툴에 적합한 형식으로 변환해야 할 수 있습니다. 특정 툴을 위한 감사 로그 변환 방법에 대한 정보는 W&B 문서의 범위를 벗어납니다.

:::tip
**감사 로그 보존:** 귀하의 조직 내 컴플라이언스, 보안 또는 리스크 팀이 감사 로그를 특정 기간 동안 보존하도록 요구하는 경우, W&B는 인스턴스 수준 버킷에서 로그를 주기적으로 장기 보존 스토리지로 전송할 것을 권장합니다. API를 사용하여 감사 로그에 엑세스하는 경우, 마지막 스크립트 실행 시점 이후 생성된 로그를 주기적으로(예: 일일 또는 며칠마다) 가져오는 간단한 스크립트를 구현하고, 이러한 로그를 분석을 위한 단기 저장소에 저장하거나 직접 장기 보존 스토리지로 전송할 수 있습니다.
:::

:::note
**감사 로그는 아직 W&B 다중 테넌트 SaaS 클라우드에서 사용할 수 없습니다.**
:::

## 감사 로그 스키마
다음 표는 감사 로그에 존재할 수 있는 모든 키를 나열합니다. 각 로그는 해당 동작과 관련된 자산만 포함하고, 다른 것들은 로그에서 생략됩니다.

| 키 | 정의 |
|---------| -------|
|timestamp               | [RFC3339 형식](https://www.rfc-editor.org/rfc/rfc3339)의 타임스탬프. 예: `2023-01-23T12:34:56Z`, `2023년 1월 23일 UTC 시간 12:34:56`을 나타냅니다.
|action                  | 사용자가 수행한 [동작](#actions).
|actor_user_id           | 있을 경우, 동작을 수행한 로그인한 사용자의 ID.
|response_code           | 동작에 대한 Http 응답 코드.
|artifact_asset          | 있을 경우, 이 아티팩트 id에 대한 동작이 수행됨.
|artifact_sequence_asset | 있을 경우, 이 아티팩트 시퀀스 id에 대한 동작이 수행됨.
|entity_asset            | 있을 경우, 이 엔티티 또는 팀 id에 대한 동작이 수행됨.
|project_asset           | 있을 경우, 이 프로젝트 id에 대한 동작이 수행됨.
|report_asset            | 있을 경우, 이 리포트 id에 대한 동작이 수행됨.
|user_asset              | 있을 경우, 이 사용자 자산에 대한 동작이 수행됨.
|cli_version             | 동작이 파이썬 SDK를 통해 이루어진 경우, 버전을 포함함.
|actor_ip                | 로그인한 사용자의 IP 주소.
|actor_email             | 있을 경우, 이 액터 이메일에 대한 동작이 수행됨.
|artifact_digest         | 있을 경우, 이 아티팩트 다이제스트에 대한 동작이 수행됨.
|artifact_qualified_name | 있을 경우, 이 아티팩트에 대한 동작이 수행됨.
|entity_name             | 있을 경우, 이 엔티티 또는 팀 이름에 대한 동작이 수행됨.
|project_name            | 있을 경우, 이 프로젝트 이름에 대한 동작이 수행됨.
|report_name             | 있을 경우, 이 리포트 이름에 대한 동작이 수행됨.
|user_email              | 있을 경우, 이 사용자 이메일에 대한 동작이 수행됨.

이메일 ID, 프로젝트, 팀 및 리포트 이름과 같은 개인 식별 정보(PII)는 API 엔드포인트 옵션을 사용하여만 사용할 수 있으며, [아래 설명된](#fetch-audit-logs-using-api)대로 끌 수 있습니다.

## API를 사용하여 감사 로그 가져오기
인스턴스 관리자는 다음 API를 사용하여 W&B 서버 인스턴스의 감사 로그를 가져올 수 있습니다:
1. 기본 엔드포인트 `<wandb-server-url>/admin/audit_logs`와 다음 URL 파라미터의 조합을 사용하여 전체 API 엔드포인트를 구성합니다:
    - `numDays` : `오늘 - numdays`부터 가장 최근까지의 로그를 가져옵니다; 기본값은 `0`이며, 즉 `오늘`에 대한 로그만 반환됩니다.
    - `anonymize` : `true`로 설정하면 모든 PII를 제거합니다; 기본값은 `false`입니다.
2. 구성된 전체 API 엔드포인트에서 HTTP GET 요청을 실행합니다. 현대적인 브라우저 내에서 직접 실행하거나, [Postman](https://www.postman.com/downloads/), [HTTPie](https://httpie.io/), cURL 코맨드 등의 툴을 사용할 수 있습니다.

W&B 서버 인스턴스 URL이 `https://mycompany.wandb.io`이고 지난 주 동안 PII 없이 감사 로그를 가져오고 싶다면, API 엔드포인트는 `https://mycompany.wandb.io?numDays=7&anonymize=true`가 됩니다.

:::note
API를 사용하여 감사 로그를 가져올 수 있는 것은 W&B 서버 [인스턴스 관리자](./manage-users.md#instance-admins)만 해당됩니다. 인스턴스 관리자가 아니거나 조직에 로그인하지 않은 경우, `HTTP 403 Forbidden` 오류가 발생합니다.
:::

API 응답에는 새 줄로 구분된 JSON 오브젝트가 포함됩니다. 오브젝트에는 스키마에 설명된 필드가 포함됩니다. 이는 인스턴스 수준 버킷으로 감사 로그 파일을 동기화할 때 사용되는 형식과 동일하며(앞서 언급한 대로 적용 가능한 경우), 이러한 경우 감사 로그는 버킷의 `/wandb-audit-logs` 디렉토리에 위치합니다.

## 동작
다음 표는 W&B에 의해 기록될 수 있는 가능한 동작을 설명합니다:

|동작 | 정의 |
|-----|-----|
| artifact:create             | 아티팩트가 생성됨.
| artifact:delete             | 아티팩트가 삭제됨.
| artifact:read               | 아티팩트가 읽힘.
| project:delete              | 프로젝트가 삭제됨.
| project:read                | 프로젝트가 읽힘.
| report:read                 | 리포트가 읽힘.
| run:delete                  | run이 삭제됨.
| run:delete_many             | 여러 run이 일괄 삭제됨.
| run:update_many             | 여러 run이 일괄 업데이트됨.
| run:stop                    | run이 중지됨.
| run:undelete_many           | 여러 run이 일괄적으로 휴지통에서 복원됨.
| run:update                  | run이 업데이트됨.
| sweep:create_agent          | 스윕 에이전트가 생성됨.
| team:invite_user            | 사용자가 팀에 초대됨.
| team:create_service_account | 팀을 위한 서비스 계정이 생성됨.
| team:create                 | 팀이 생성됨.
| team:uninvite               | 사용자 또는 서비스 계정이 팀에서 초대 취소됨.
| team:delete                 | 팀이 삭제됨.
| user:create                 | 사용자가 생성됨.
| user:delete_api_key         | 사용자의 API 키가 삭제됨.
| user:deactivate             | 사용자가 비활성화됨.
| user:create_api_key         | 사용자를 위한 API 키가 생성됨.
| user:permanently_delete     | 사용자가 영구적으로 삭제됨.
| user:reactivate             | 사용자가 재활성화됨.
| user:update                 | 사용자가 업데이트됨.
| user:read                   | 사용자 프로필이 읽힘.
| user:login                  | 사용자가 로그인함.
| user:initiate_login         | 사용자가 로그인을 시작함.
| user:logout                 | 사용자가 로그아웃함.