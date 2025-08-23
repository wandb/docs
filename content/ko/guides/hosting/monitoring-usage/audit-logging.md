---
title: 감사 로그로 사용자 활동 추적하기
menu:
  default:
    identifier: ko-guides-hosting-monitoring-usage-audit-logging
    parent: monitoring-and-usage
weight: 1
---

W&B 감사 로그를 사용하여 조직 내 사용자 활동을 추적하고, 엔터프라이즈 거버넌스 요구 사항을 준수할 수 있습니다. 감사 로그는 JSON 포맷으로 제공됩니다. 자세한 내용은 [Audit log schema]({{< relref path="#audit-log-schema" lang="ko" >}})를 참고하세요.

감사 로그 엑세스 방법은 사용 중인 W&B 플랫폼 배포 유형에 따라 다릅니다:

| W&B 플랫폼 배포 유형 | 감사 로그 엑세스 방법 |
|----------------------------|--------------------------------|
| [Self-Managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}) | 10분마다 인스턴스 수준 버킷에 동기화됩니다. [API]({{< relref path="#fetch-audit-logs-using-api" lang="ko" >}})를 통해서도 확인할 수 있습니다. |
| [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) 및 [보안 스토리지 커넥터(BYOB)]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ko" >}}) 사용 시 | 10분마다 인스턴스 수준 버킷(BYOB)에 동기화됩니다. [API]({{< relref path="#fetch-audit-logs-using-api" lang="ko" >}})를 통해서도 확인할 수 있습니다. |
| [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) 및 W&B 관리 스토리지(BYOB 미사용) | [API]({{< relref path="#fetch-audit-logs-using-api" lang="ko" >}})를 통해서만 확인할 수 있습니다. |
| [Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}}) | 엔터프라이즈 플랜에서만 제공되며, [API]({{< relref path="#fetch-audit-logs-using-api" lang="ko" >}})를 통해서만 확인할 수 있습니다.

감사 로그를 가져온 뒤 [Pandas](https://pandas.pydata.org/docs/index.html), [Amazon Redshift](https://aws.amazon.com/redshift/), [Google BigQuery](https://cloud.google.com/bigquery), [Microsoft Fabric](https://www.microsoft.com/microsoft-fabric) 등 다양한 분석 툴로 분석할 수 있습니다. 일부 감사 로그 분석 툴은 JSON을 지원하지 않을 수 있으니, 사용하려는 분석 툴의 가이드를 참고해 JSON 포맷의 로그를 적절히 변환하세요.

{{% alert title="Audit log retention" %}}
특정 기간 동안 감사 로그를 보관해야 하는 경우, 저장 버킷 또는 감사 로깅 API를 사용하여 주기적으로 로그를 장기 보관 스토리지로 이전하는 것을 권장합니다.

[Health Insurance Portability and Accountability Act of 1996 (HIPAA)](https://www.hhs.gov/hipaa/for-professionals/index.html) 규제를 받는 경우, 감사 로그는 최소 6년간 삭제되거나 변경될 수 없는 환경에 보관되어야 합니다. HIPAA 인증 [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) 인스턴스에서 [BYOB]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ko" >}})를 사용할 경우 장기 보관 스토리지를 포함한 스토리지 관리에 대한 guardrail 구성이 필요합니다.
{{% /alert %}}

## Audit log schema
이 표는 감사 로그 항목에 포함될 수 있는 모든 키를 알파벳 순으로 보여줍니다. 동작 및 상황에 따라 각 로그 항목에는 가능한 필드의 일부만 포함될 수 있습니다.

| Key | 정의 |
|---------| -------|
|`action`                  | 이벤트의 [action]({{< relref path="#actions" lang="ko" >}}).
|`actor_email`             | 해당 동작을 수행한 사용자의 이메일 어드레스(해당되는 경우).
|`actor_ip`                | 해당 동작을 수행한 사용자의 IP 어드레스.
|`actor_user_id`           | 동작을 수행한 로그인된 사용자의 ID(해당되는 경우).
|`artifact_asset`          | 동작과 연관된 Artifact ID(해당되는 경우).
|`artifact_digest`         | 동작과 연관된 Artifact digest(해당되는 경우).
|`artifact_qualified_name` | 동작과 연관된 Artifact의 전체 이름(해당되는 경우).
|`artifact_sequence_asset` | 동작과 연관된 Artifact sequence ID(해당되는 경우).
|`cli_version`             | 동작을 일으킨 Python SDK 버전(해당되는 경우).
|`entity_asset`            | 동작과 연관된 Entity 또는 팀 ID(해당되는 경우).
|`entity_name`             | 동작과 연관된 Entity 또는 팀 이름(해당되는 경우).
|`project_asset`           | 동작과 연관된 Project(해당되는 경우).
|`project_name`            | 동작과 연관된 Project 이름(해당되는 경우).
|`report_asset`            | 동작과 연관된 Report ID(해당되는 경우).
|`report_name`             | 동작과 연관된 Report 이름(해당되는 경우).
|`response_code`           | 동작에 대한 HTTP 응답 코드(해당되는 경우).
|`timestamp`               | 이벤트 시간([RFC3339 format](https://www.rfc-editor.org/rfc/rfc3339)). 예: `2023-01-23T12:34:56Z`는 2023년 1월 23일 12시 34분 56초 UTC를 의미.
|`user_asset`              | 동작의 영향을 받은 user asset(동작을 수행한 사용자가 아니라, 영향을 받은 사용자)(해당되는 경우).
|`user_email`              | 동작의 영향을 받은 사용자의 이메일 어드레스(동작을 수행한 사용자가 아니라, 영향을 받은 사용자)(해당되는 경우).

### 개인 식별 정보(PII)

개인 식별 정보(PII)는 이메일 어드레스, 프로젝트, 팀, Report 명 등이며, 이는 API 엔드포인트 옵션을 사용할 때만 조회할 수 있습니다.
- [Self-Managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}) 및 
  [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) 환경의 경우, 조직 관리자가 [PII 제외]({{< relref path="#exclude-pii" lang="ko" >}}) 옵션으로 감사 로그를 가져올 수 있습니다.
- [Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}}) 환경에서는 항상 감사 로그와 관련된 필드(PII 포함)가 반환되며, 이는 설정에서 변경할 수 없습니다.

## Fetch audit logs
조직 또는 인스턴스 관리자는 Audit Logging API의 `audit_logs/` 엔드포인트를 통해 W&B 인스턴스의 감사 로그를 가져올 수 있습니다.

{{% alert %}}
- 관리자가 아닌 사용자가 감사 로그를 요청하면, HTTP `403` 에러가 발생하며 엑세스 거부를 의미합니다.

- 여러 엔터프라이즈 [Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}}) 조직의 관리자인 경우, 감사 로깅 API 요청이 어느 조직으로 보내질지 설정해야 합니다. 프로필 이미지를 클릭한 후 **User Settings**에 진입하세요. 해당 설정 명칭은 **Default API organization**입니다.
{{% /alert %}}

1. 본인의 인스턴스에 맞는 API 엔드포인트를 확인하세요:

    - [Self-Managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}): `<wandb-platform-url>/admin/audit_logs`
    - [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}): `<wandb-platform-url>/admin/audit_logs`
    - [Multi-tenant Cloud (Enterprise 필수)]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}}): `https://api.wandb.ai/audit_logs`

    이후 단계에서 `<API-endpoint>`를 해당 엔드포인트로 바꿔 사용하세요.
1. 베이스 엔드포인트에 URL 파라미터를 추가해 전체 API 엔드포인트를 만듭니다:
    - `anonymize`: `true`로 설정하면 모든 PII를 제거합니다. 기본값은 `false`입니다. [감사 로그 가져올 때 PII 제외하기]({{< relref path="#exclude-pii" lang="ko" >}}) 참고. Multi-tenant Cloud에서는 지원하지 않습니다.
    - `numDays`: `today - numdays` 시점부터 최신까지의 로그를 가져옵니다. 기본값은 `0`으로 `today`에 해당하는 로그만 반환됩니다. Multi-tenant Cloud 환경에서는 최대 7일까지 과거 로그를 조회할 수 있습니다.
    - `startDate`: `YYYY-MM-DD` 포맷의 선택적 날짜. [Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}})에서만 지원됩니다.

      `startDate`와 `numDays`의 동작:
        - 둘 다 지정하면, `startDate`부터 `startDate + numDays`까지의 로그를 반환합니다.
        - `startDate`는 없이 `numDays`만 지정 시, 오늘 기준으로 `numDays`만큼 조회합니다.
        - 둘 다 생략 시, 오늘자 로그만 반환합니다.

1. 완성된 API 엔드포인트로 웹브라우저, [Postman](https://www.postman.com/downloads/), [HTTPie](https://httpie.io/), cURL 등으로 HTTP `GET` 요청을 실행합니다.

API 응답은 각 줄마다 하나의 JSON 오브젝트가 포함된 형태입니다. 오브젝트에는 [schema]({{< relref path="#audit-log-schemag" lang="ko" >}})에 명시된 필드가 포함되며, 인스턴스 버킷에 동기화될 때와 동일합니다. 이 경우 감사 로그는 버킷의 `/wandb-audit-logs` 디렉토리에 위치하게 됩니다.

### Basic 인증 사용하기
API 키로 감사 로그 API를 엑세스할 때 Basic 인증을 사용하려면 HTTP 요청의 `Authorization` 헤더에 `Basic` 뒤에 공백, 그리고 `username:API-KEY` 형식의 base-64 인코딩된 문자열을 입력해야 합니다. 즉, username과 API 키를 `:`로 구분하여 연결한 뒤, 그 결과를 base-64로 인코딩합니다. 예를 들어, `demo:p@55w0rd`로 인증하려면 헤더는 `Authorization: Basic ZGVtbzpwQDU1dzByZA==`와 같아야 합니다.

### 감사 로그 가져올 때 PII 제외하기 {#exclude-pii}
[Self-Managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}) 및 [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) 환경에서는 조직 또는 인스턴스 관리자가 감사 로그에서 PII를 제외시킬 수 있습니다. [Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}})에서는 감사 로그 관련 필드(PII 포함)가 항상 반환되고, 설정을 변경할 수 없습니다.

PII를 제외하려면 `anonymize=true` URL 파라미터를 추가하세요. 예를 들어, W&B 인스턴스 URL이 `https://mycompany.wandb.io`이고 지난 주 동안의 사용자 활동 감사 로그에서 PII를 제외하려는 경우:

```text
https://mycompany.wandb.io/admin/audit_logs?numDays=7&anonymize=true.
```

## Actions
W&B에서 기록할 수 있는 모든 동작을 알파벳 순서로 나열합니다.

|Action | 정의 |
|-----|-----|
| `artifact:create`             | Artifact가 생성됨.
| `artifact:delete   `          | Artifact가 삭제됨.
| `artifact:read`               | Artifact가 조회됨.
| `project:delete`              | Project가 삭제됨.
| `project:read`                | Project가 조회됨.
| `report:read`                 | Report가 조회됨. <sup><a href="#1">1</a></sup>
| `run:delete_many`             | 여러 개의 run이 삭제됨.
| `run:delete`                  | run이 삭제됨.
| `run:stop`                    | run이 중지됨.
| `run:undelete_many`           | 여러 개의 run이 휴지통에서 복원됨.
| `run:update_many`             | 여러 개의 run이 업데이트됨.
| `run:update`                  | run이 업데이트됨.
| `sweep:create_agent`          | Sweep agent가 생성됨.
| `team:create_service_account` | 팀을 위한 서비스 계정이 생성됨.
| `team:create`                 | 팀이 생성됨.
| `team:delete`                 | 팀이 삭제됨.
| `team:invite_user`            | 사용자가 팀에 초대됨.
| `team:uninvite`               | 사용자 또는 서비스 계정이 팀에서 초대해제됨.
| `user:create_api_key`         | 사용자의 API 키가 생성됨. <sup><a href="#1">1</a></sup>
| `user:create`                 | 사용자가 생성됨. <sup><a href="#1">1</a></sup>
| `user:deactivate`             | 사용자가 비활성화됨. <sup><a href="#1">1</a></sup>
| `user:delete_api_key`         | 사용자의 API 키가 삭제됨. <sup><a href="#1">1</a></sup>
| `user:initiate_login`         | 사용자가 로그인을 시작함. <sup><a href="#1">1</a></sup>
| `user:login`                  | 사용자가 로그인함. <sup><a href="#1">1</a></sup>
| `user:logout`                 | 사용자가 로그아웃함. <sup><a href="#1">1</a></sup>
| `user:permanently_delete`     | 사용자가 영구 삭제됨. <sup><a href="#1">1</a></sup>
| `user:reactivate`             | 사용자가 재활성화됨. <sup><a href="#1">1</a></sup>
| `user:read`                   | 사용자 프로필이 조회됨. <sup><a href="#1">1</a></sup>
| `user:update`                 | 사용자가 업데이트됨. <sup><a href="#1">1</a></sup>

<a id="1">1</a>: [Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}})에서는 다음의 경우에 감사 로그가 수집되지 않습니다.
- 공개 또는 오픈 Project
- `report:read` 액션
- 특정 조직에 속하지 않은 User 액션