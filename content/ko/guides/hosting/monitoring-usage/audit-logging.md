---
title: Track user activity with audit logs
menu:
  default:
    identifier: ko-guides-hosting-monitoring-usage-audit-logging
    parent: monitoring-and-usage
weight: 1
---

W&B 감사 로그를 사용하여 조직 내 사용자 활동을 추적하고 엔터프라이즈 거버넌스 요구 사항을 준수하십시오. 감사 로그는 JSON 형식으로 제공됩니다. 감사 로그에 엑세스하는 방법은 W&B 플랫폼 배포 유형에 따라 다릅니다.

| W&B 플랫폼 배포 유형 | 감사 로그 엑세스 메커니즘 |
|----------------------------|--------------------------------|
| [자체 관리]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}) | 10분마다 인스턴스 수준 버킷에 동기화됩니다. [API]({{< relref path="#fetch-audit-logs-using-api" lang="ko" >}})를 사용하여 사용할 수도 있습니다. |
| [전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) ([보안 스토리지 커넥터(BYOB) 포함]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ko" >}})) | 10분마다 인스턴스 수준 버킷(BYOB)에 동기화됩니다. [API]({{< relref path="#fetch-audit-logs-using-api" lang="ko" >}})를 사용하여 사용할 수도 있습니다. |
| W&B 관리 스토리지([전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) BYOB 제외) | [API]({{< relref path="#fetch-audit-logs-using-api" lang="ko" >}})를 통해서만 사용할 수 있습니다. |

{{% alert %}}
감사 로그는 [SaaS Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}})에서는 사용할 수 없습니다.
{{% /alert %}}

감사 로그에 엑세스한 후에는 [Pandas](https://pandas.pydata.org/docs/index.html), [Amazon Redshift](https://aws.amazon.com/redshift/), [Google BigQuery](https://cloud.google.com/bigquery), [Microsoft Fabric](https://www.microsoft.com/en-us/microsoft-fabric) 등과 같은 선호하는 툴을 사용하여 분석하십시오. 분석하기 전에 JSON 형식의 감사 로그를 툴과 관련된 형식으로 변환해야 할 수도 있습니다. 특정 툴에 대한 감사 로그 변환 방법에 대한 정보는 W&B 문서의 범위를 벗어납니다.

{{% alert %}}
**감사 로그 보존:** 조직의 규정 준수, 보안 또는 위험 관리 팀에서 특정 기간 동안 감사 로그를 보존해야 하는 경우 W&B는 인스턴스 수준 버킷에서 장기 보존 스토리지로 로그를 주기적으로 전송하는 것이 좋습니다. 대신 API를 사용하여 감사 로그에 엑세스하는 경우 마지막 스크립트 실행 이후 생성되었을 수 있는 로그를 가져오고 분석을 위해 단기 스토리지에 저장하거나 장기 보존 스토리지로 직접 전송하기 위해 주기적으로(예: 매일 또는 며칠마다) 실행되는 간단한 스크립트를 구현할 수 있습니다.
{{% /alert %}}

HIPAA 규정 준수를 위해서는 최소 6년 동안 감사 로그를 보존해야 합니다. HIPAA를 준수하는 [전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) 인스턴스([BYOB]({{< relref path="/guides/hosting/data-security/secure-storage-connector.md" lang="ko" >}}) 포함)의 경우 내부 또는 외부 사용자가 필수 보존 기간이 끝나기 전에 감사 로그를 삭제할 수 없도록 장기 보존 스토리지를 포함한 관리형 스토리지에 대한 보호 장치를 구성해야 합니다.

## 감사 로그 스키마
다음 표에는 감사 로그에 있을 수 있는 모든 키가 나열되어 있습니다. 각 로그에는 해당 작업과 관련된 자산만 포함되어 있으며 다른 자산은 로그에서 생략됩니다.

| 키 | 정의 |
|---------| -------|
|timestamp | [RFC3339 형식](https://www.rfc-editor.org/rfc/rfc3339)의 타임스탬프입니다. 예: `2023-01-23T12:34:56Z`는 2023년 1월 23일 `12:34:56 UTC` 시간을 나타냅니다.
|action | 사용자가 수행한 [작업]({{< relref path="#actions" lang="ko" >}})입니다.
|actor_user_id | 있는 경우 작업을 수행한 로그인한 사용자의 ID입니다.
|response_code | 작업에 대한 Http 응답 코드입니다.
|artifact_asset | 있는 경우 이 artifact id에 대해 작업이 수행되었습니다.
|artifact_sequence_asset | 있는 경우 이 artifact sequence id에 대해 작업이 수행되었습니다.
|entity_asset | 있는 경우 이 entity 또는 team id에 대해 작업이 수행되었습니다.
|project_asset | 있는 경우 이 project id에 대해 작업이 수행되었습니다.
|report_asset | 있는 경우 이 report id에 대해 작업이 수행되었습니다.
|user_asset | 있는 경우 이 user asset에 대해 작업이 수행되었습니다.
|cli_version | 작업을 Python SDK를 통해 수행한 경우 버전을 포함합니다.
|actor_ip | 로그인한 사용자의 IP 어드레스입니다.
|actor_email | 있는 경우 이 actor email에 대해 작업이 수행되었습니다.
|artifact_digest | 있는 경우 이 artifact digest에 대해 작업이 수행되었습니다.
|artifact_qualified_name | 있는 경우 이 artifact에 대해 작업이 수행되었습니다.
|entity_name | 있는 경우 이 entity 또는 team 이름에 대해 작업이 수행되었습니다.
|project_name | 있는 경우 이 project 이름에 대해 작업이 수행되었습니다.
|report_name | 있는 경우 이 report 이름에 대해 작업이 수행되었습니다.
|user_email | 있는 경우 이 user email에 대해 작업이 수행되었습니다.

이메일 ID, Projects, Teams 및 Reports 이름과 같은 개인 식별 정보(PII)는 API 엔드포인트 옵션을 통해서만 사용할 수 있으며 [아래 설명된 대로]({{< relref path="#fetch-audit-logs-using-api" lang="ko" >}}) 끌 수 있습니다.

## API를 사용하여 감사 로그 가져오기
인스턴스 관리자는 다음 API를 사용하여 W&B 인스턴스에 대한 감사 로그를 가져올 수 있습니다.
1. 기본 엔드포인트 `<wandb-platform-url>/admin/audit_logs`와 다음 URL 파라미터를 조합하여 전체 API 엔드포인트를 구성합니다.
    - `numDays`: 로그는 `오늘 - numdays`부터 가장 최근까지 가져옵니다. 기본값은 `0`이며, `오늘`에 대한 로그만 반환합니다.
    - `anonymize`: `true`로 설정하면 모든 PII를 제거합니다. 기본값은 `false`입니다.
2. 최신 브라우저 내에서 직접 실행하거나 [Postman](https://www.postman.com/downloads/), [HTTPie](https://httpie.io/), cURL 코맨드 등과 같은 툴을 사용하여 구성된 전체 API 엔드포인트에서 HTTP GET 요청을 실행합니다.

조직 또는 인스턴스 관리자는 API 키로 기본 인증을 사용하여 감사 로그 API에 엑세스할 수 있습니다. HTTP 요청의 `Authorization` 헤더를 문자열 `Basic` 다음에 공백, 그런 다음 `username:API-KEY` 형식으로 base-64로 인코딩된 문자열로 설정합니다. 즉, 사용자 이름과 API 키를 `:` 문자로 구분된 값으로 바꾸고 결과를 base-64로 인코딩합니다. 예를 들어 `demo:p@55w0rd`로 인증하려면 헤더는 `Authorization: Basic ZGVtbzpwQDU1dzByZA==`여야 합니다.

W&B 인스턴스 URL이 `https://mycompany.wandb.io`이고 지난 주 동안의 사용자 활동에 대한 PII 없이 감사 로그를 가져오려면 API 엔드포인트 `https://mycompany.wandb.io/admin/audit_logs?numDays=7&anonymize=true`를 사용해야 합니다.

{{% alert %}}
W&B [인스턴스 관리자]({{< relref path="/guides/hosting/iam/access-management/" lang="ko" >}})만 API를 사용하여 감사 로그를 가져올 수 있습니다. 인스턴스 관리자가 아니거나 조직에 로그인하지 않은 경우 `HTTP 403 Forbidden` 오류가 발생합니다.
{{% /alert %}}

API 응답에는 줄 바꿈으로 구분된 JSON 오브젝트가 포함되어 있습니다. 오브젝트에는 스키마에 설명된 필드가 포함됩니다. 감사 로그 파일을 인스턴스 수준 버킷에 동기화할 때 사용되는 것과 동일한 형식입니다(앞서 언급한 대로 해당되는 경우). 이러한 경우 감사 로그는 버킷의 `/wandb-audit-logs` 디렉토리에 있습니다.

## 작업
다음 표에서는 W&B에서 기록할 수 있는 가능한 작업을 설명합니다.

|작업 | 정의 |
|-----|-----|
| artifact:create | Artifact가 생성되었습니다.
| artifact:delete | Artifact가 삭제되었습니다.
| artifact:read | Artifact가 읽혔습니다.
| project:delete | Project가 삭제되었습니다.
| project:read | Project가 읽혔습니다.
| report:read | Report가 읽혔습니다.
| run:delete | Run이 삭제되었습니다.
| run:delete_many | Runs가 일괄적으로 삭제되었습니다.
| run:update_many | Runs가 일괄적으로 업데이트되었습니다.
| run:stop | Run이 중지되었습니다.
| run:undelete_many | Runs가 일괄적으로 휴지통에서 다시 가져왔습니다.
| run:update | Run이 업데이트되었습니다.
| sweep:create_agent | 스윕 에이전트가 생성되었습니다.
| team:invite_user | User가 Team에 초대되었습니다.
| team:create_service_account | Team에 대한 서비스 계정이 생성되었습니다.
| team:create | Team이 생성되었습니다.
| team:uninvite | User 또는 서비스 계정이 Team에서 초대 취소되었습니다.
| team:delete | Team이 삭제되었습니다.
| user:create | User가 생성되었습니다.
| user:delete_api_key | User에 대한 API 키가 삭제되었습니다.
| user:deactivate | User가 비활성화되었습니다.
| user:create_api_key | User에 대한 API 키가 생성되었습니다.
| user:permanently_delete | User가 영구적으로 삭제되었습니다.
| user:reactivate | User가 다시 활성화되었습니다.
| user:update | User가 업데이트되었습니다.
| user:read | User 프로필이 읽혔습니다.
| user:login | User가 로그인했습니다.
| user:initiate_login | User가 로그인을 시작했습니다.
| user:logout | User가 로그아웃했습니다.
