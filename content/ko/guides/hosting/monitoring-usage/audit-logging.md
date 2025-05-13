---
title: Track user activity with audit logs
menu:
  default:
    identifier: ko-guides-hosting-monitoring-usage-audit-logging
    parent: monitoring-and-usage
weight: 1
---

W&B 감사 로그를 사용하여 조직 내 사용자 활동을 추적하고 엔터프라이즈 거버넌스 요구 사항을 준수하십시오. 감사 로그는 JSON 형식으로 제공됩니다. [감사 로그 스키마]({{< relref path="#audit-log-schema" lang="ko" >}})를 참조하십시오.

감사 로그에 액세스하는 방법은 W&B 플랫폼 배포 유형에 따라 다릅니다.

| W&B Platform 배포 유형 | 감사 로그 엑세스 메커니즘 |
|----------------------------|--------------------------------|
| [자체 관리]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}) | 10분마다 인스턴스 수준 버킷에 동기화됩니다. [API]({{< relref path="#fetch-audit-logs-using-api" lang="ko" >}})를 사용하여 사용할 수도 있습니다. |
| [전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) ([보안 스토리지 커넥터 (BYOB)]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ko" >}})) | 10분마다 인스턴스 수준 버킷 (BYOB)에 동기화됩니다. [API]({{< relref path="#fetch-audit-logs-using-api" lang="ko" >}})를 사용하여 사용할 수도 있습니다. |
| W&B 관리 스토리지 (BYOB 없음)가 있는 [전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) | [API]({{< relref path="#fetch-audit-logs-using-api" lang="ko" >}})를 통해서만 사용할 수 있습니다. |
| [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}}) | Enterprise 요금제에서만 사용할 수 있습니다. [API]({{< relref path="#fetch-audit-logs-using-api" lang="ko" >}})를 통해서만 사용할 수 있습니다.

감사 로그를 가져온 후에는 [Pandas](https://pandas.pydata.org/docs/index.html), [Amazon Redshift](https://aws.amazon.com/redshift/), [Google BigQuery](https://cloud.google.com/bigquery) 또는 [Microsoft Fabric](https://www.microsoft.com/microsoft-fabric)과 같은 툴을 사용하여 분석할 수 있습니다. 일부 감사 로그 분석 툴은 JSON을 지원하지 않습니다. 분석 전에 JSON 형식의 감사 로그를 변환하기 위한 지침 및 요구 사항은 분석 툴 설명서를 참조하십시오.

{{% alert title="감사 로그 보존" %}}
특정 기간 동안 감사 로그를 보존해야 하는 경우 W&B는 스토리지 버킷 또는 감사 로깅 API를 사용하여 장기 스토리지로 로그를 주기적으로 전송하는 것이 좋습니다.

[1996년 건강 보험 양도 및 책임에 관한 법률 (HIPAA)](https://www.hhs.gov/hipaa/for-professionals/index.html)의 적용을 받는 경우 감사 로그는 의무 보존 기간이 끝나기 전에 내부 또는 외부 행위자가 삭제하거나 수정할 수 없는 환경에서 최소 6년 동안 보존해야 합니다. [BYOB]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ko" >}})가 있는 HIPAA 규정 준수 [전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) 인스턴스의 경우 장기 보존 스토리지를 포함하여 관리형 스토리지에 대한 가드레일을 구성해야 합니다.
{{% /alert %}}

## 감사 로그 스키마
이 표는 감사 로그 항목에 나타날 수 있는 모든 키를 알파벳순으로 보여줍니다. 작업 및 상황에 따라 특정 로그 항목에는 가능한 필드의 서브셋만 포함될 수 있습니다.

| 키 | 정의 |
|---------| -------|
|`action`                  | 이벤트의 [action]({{< relref path="#actions" lang="ko" >}}).
|`actor_email`             | 해당되는 경우, 작업을 시작한 사용자의 이메일 어드레스.
|`actor_ip`                | 작업을 시작한 사용자의 IP 어드레스.
|`actor_user_id`           | 해당되는 경우, 작업을 수행한 로그인한 사용자의 ID.
|`artifact_asset`          | 해당되는 경우, 작업과 관련된 artifact ID.
|`artifact_digest`         | 해당되는 경우, 작업과 관련된 artifact 다이제스트.
|`artifact_qualified_name` | 해당되는 경우, 작업과 관련된 artifact의 전체 이름.
|`artifact_sequence_asset` | 해당되는 경우, 작업과 관련된 artifact 시퀀스 ID.
|`cli_version`             | 해당되는 경우, 작업을 시작한 Python SDK의 버전.
|`entity_asset`            | 해당되는 경우, 작업과 관련된 entity 또는 팀 ID.
|`entity_name`             | 해당되는 경우, entity 또는 팀 이름.
|`project_asset`           | 작업과 관련된 project.
|`project_name`            | 작업과 관련된 project 이름.
|`report_asset`            | 작업과 관련된 report ID.
|`report_name`             | 작업과 관련된 report 이름.
|`response_code`           | 해당되는 경우, 작업에 대한 HTTP 응답 코드.
|`timestamp`               | [RFC3339 형식](https://www.rfc-editor.org/rfc/rfc3339)의 이벤트 시간. 예를 들어, `2023-01-23T12:34:56Z`는 2023년 1월 23일 12:34:56 UTC를 나타냅니다.
|`user_asset`              | 해당되는 경우, 작업이 영향을 미치는 user 에셋 (작업을 수행하는 user가 아님).
|`user_email`              | 해당되는 경우, 작업이 영향을 미치는 user의 이메일 어드레스 (작업을 수행하는 user의 이메일 어드레스가 아님).

### 개인 식별 정보 (PII)

이메일 어드레스, project 이름, 팀 및 report와 같은 개인 식별 정보 (PII)는 API 엔드포인트 옵션을 통해서만 사용할 수 있습니다.
- [자체 관리]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}) 및
  [전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}})의 경우 조직 관리자는 감사 로그를 가져올 때 [PII를 제외]({{< relref path="#exclude-pii" lang="ko" >}})할 수 있습니다.
- [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}})의 경우 API 엔드포인트는 항상 PII를 포함하여 감사 로그에 대한 관련 필드를 반환합니다. 이는 구성할 수 없습니다.

## 감사 로그 가져오기
조직 또는 인스턴스 관리자는 `audit_logs/` 엔드포인트에서 감사 로깅 API를 사용하여 W&B 인스턴스에 대한 감사 로그를 가져올 수 있습니다.

{{% alert %}}
- 관리자 이외의 사용자가 감사 로그를 가져오려고 하면 엑세스가 거부되었음을 나타내는 HTTP `403` 오류가 발생합니다.

- 여러 Enterprise [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}}) 조직의 관리자인 경우 감사 로깅 API 요청이 전송되는 조직을 구성해야 합니다. 프로필 이미지를 클릭한 다음 **User Settings**를 클릭합니다. 설정 이름은 **Default API organization**입니다.
{{% /alert %}}

1. 인스턴스에 적합한 API 엔드포인트를 결정합니다.

    - [자체 관리]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}): `<wandb-platform-url>/admin/audit_logs`
    - [전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}): `<wandb-platform-url>/admin/audit_logs`
    - [SaaS Cloud (Enterprise 필요)]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}}): `https://api.wandb.ai/audit_logs`

    다음 단계에서 `<API-endpoint>`를 API 엔드포인트로 바꿉니다.
2. 기본 엔드포인트에서 전체 API 엔드포인트를 구성하고 선택적으로 URL 파라미터를 포함합니다.
    - `anonymize`: `true`로 설정하면 모든 PII를 제거합니다. 기본값은 `false`입니다. [감사 로그를 가져올 때 PII 제외]({{< relref path="#exclude-pii" lang="ko" >}})를 참조하십시오. SaaS Cloud에서는 지원되지 않습니다.
    - `numDays`: 로그는 `today - numdays`부터 가장 최근까지 가져옵니다. 기본값은 `0`이며, `today`에 대한 로그만 반환합니다. SaaS Cloud의 경우 최대 7일 전의 감사 로그를 가져올 수 있습니다.
    - `startDate`: `YYYY-MM-DD` 형식의 선택적 날짜입니다. [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}})에서만 지원됩니다.

      `startDate`와 `numDays`는 상호 작용합니다.
        - `startDate`와 `numDays`를 모두 설정하면 `startDate`부터 `startDate` + `numDays`까지 로그가 반환됩니다.
        - `startDate`를 생략했지만 `numDays`를 포함하면 `today`부터 `numDays`까지 로그가 반환됩니다.
        - `startDate`와 `numDays`를 모두 설정하지 않으면 `today`에 대한 로그만 반환됩니다.

3. 웹 브라우저 또는 [Postman](https://www.postman.com/downloads/), [HTTPie](https://httpie.io/) 또는 cURL과 같은 툴을 사용하여 구성된 정규화된 전체 API 엔드포인트에서 HTTP `GET` 요청을 실행합니다.

API 응답에는 줄 바꿈으로 구분된 JSON 오브젝트가 포함됩니다. 오브젝트에는 감사 로그가 인스턴스 수준 버킷에 동기화될 때와 마찬가지로 [스키마]({{< relref path="#audit-log-schemag" lang="ko" >}})에 설명된 필드가 포함됩니다. 이러한 경우 감사 로그는 버킷의 `/wandb-audit-logs` 디렉토리에 있습니다.

### 기본 인증 사용
API 키로 기본 인증을 사용하여 감사 로그 API에 액세스하려면 HTTP 요청의 `Authorization` 헤더를 `Basic` 문자열로 설정하고 공백을 추가한 다음 `username:API-KEY` 형식으로 base-64로 인코딩된 문자열을 설정합니다. 즉, 사용자 이름과 API 키를 `:` 문자로 구분된 값으로 바꾸고 결과를 base-64로 인코딩합니다. 예를 들어 `demo:p@55w0rd`로 인증하려면 헤더는 `Authorization: Basic ZGVtbzpwQDU1dzByZA==`여야 합니다.

### 감사 로그를 가져올 때 PII 제외 {#exclude-pii}
[자체 관리]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}) 및 [전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}})의 경우 W&B 조직 또는 인스턴스 관리자는 감사 로그를 가져올 때 PII를 제외할 수 있습니다. [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}})의 경우 API 엔드포인트는 항상 PII를 포함하여 감사 로그에 대한 관련 필드를 반환합니다. 이는 구성할 수 없습니다.

PII를 제외하려면 `anonymize=true` URL 파라미터를 전달합니다. 예를 들어 W&B 인스턴스 URL이 `https://mycompany.wandb.io`이고 지난 주 동안의 사용자 활동에 대한 감사 로그를 가져오고 PII를 제외하려면 다음과 같은 API 엔드포인트를 사용합니다.

```text
https://mycompany.wandb.io/admin/audit_logs?numDays=7&anonymize=true.
```

## Actions
이 표는 W&B에서 기록할 수 있는 가능한 actions를 알파벳순으로 설명합니다.

|Action | 정의 |
|-----|-----|
| `artifact:create`             | Artifact가 생성되었습니다.
| `artifact:delete   `          | Artifact가 삭제되었습니다.
| `artifact:read`               | Artifact가 읽혔습니다.
| `project:delete`              | Project가 삭제되었습니다.
| `project:read`                | Project가 읽혔습니다.
| `report:read`                 | Report가 읽혔습니다. <sup><a href="#1">1</a></sup>
| `run:delete_many`             | Run 배치가 삭제되었습니다.
| `run:delete`                  | Run이 삭제되었습니다.
| `run:stop`                    | Run이 중지되었습니다.
| `run:undelete_many`           | 휴지통에서 Run 배치가 복원되었습니다.
| `run:update_many`             | Run 배치가 업데이트되었습니다.
| `run:update`                  | Run이 업데이트되었습니다.
| `sweep:create_agent`          | 스윕 에이전트가 생성되었습니다.
| `team:create_service_account` | 팀에 대한 서비스 계정이 생성되었습니다.
| `team:create`                 | 팀이 생성되었습니다.
| `team:delete`                 | 팀이 삭제되었습니다.
| `team:invite_user`            | User가 팀에 초대되었습니다.
| `team:uninvite`               | User 또는 서비스 계정이 팀에서 초대 취소되었습니다.
| `user:create_api_key`         | User에 대한 API 키가 생성되었습니다. <sup><a href="#1">1</a></sup>
| `user:create`                 | User가 생성되었습니다. <sup><a href="#1">1</a></sup>
| `user:deactivate`             | User가 비활성화되었습니다. <sup><a href="#1">1</a></sup>
| `user:delete_api_key`         | User에 대한 API 키가 삭제되었습니다. <sup><a href="#1">1</a></sup>
| `user:initiate_login`         | User가 로그인을 시작했습니다. <sup><a href="#1">1</a></sup>
| `user:login`                  | User가 로그인했습니다. <sup><a href="#1">1</a></sup>
| `user:logout`                 | User가 로그아웃했습니다. <sup><a href="#1">1</a></sup>
| `user:permanently_delete`     | User가 영구적으로 삭제되었습니다. <sup><a href="#1">1</a></sup>
| `user:reactivate`             | User가 다시 활성화되었습니다. <sup><a href="#1">1</a></sup>
| `user:read`                   | User 프로필이 읽혔습니다. <sup><a href="#1">1</a></sup>
| `user:update`                 | User가 업데이트되었습니다. <sup><a href="#1">1</a></sup>

<a id="1">1</a>: [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}})에서 감사 로그는 다음에 대해 수집되지 않습니다.
- 공개 또는 퍼블릭 프로젝트.
- `report:read` action.
- 특정 조직에 연결되지 않은 `User` actions.
