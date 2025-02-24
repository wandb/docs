---
title: Advanced IAM configuration
menu:
  default:
    identifier: ko-guides-hosting-iam-advanced_env_vars
    parent: identity-and-access-management-iam
---

기본적인 [환경 변수]({{< relref path="../env-vars.md" lang="ko" >}}) 외에도, 환경 변수를 사용하여 [전용 클라우드]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) 또는 [자체 관리]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}) 인스턴스에 대한 IAM 옵션을 구성할 수 있습니다.

IAM 요구 사항에 따라 인스턴스에 대해 다음 환경 변수 중 하나를 선택하십시오.

| 환경 변수 | 설명 |
|----------------------|-------------|
| DISABLE_SSO_PROVISIONING | W&B 인스턴스에서 사용자 자동 프로비저닝을 끄려면 `true`로 설정하십시오. |
| SESSION_LENGTH | 기본 사용자 세션 만료 시간을 변경하려면 이 변수를 원하는 시간 수로 설정하십시오. 예를 들어 세션 만료 시간을 24시간으로 구성하려면 SESSION_LENGTH를 `24`로 설정하십시오. 기본값은 720시간입니다. |
| GORILLA_ENABLE_SSO_GROUP_CLAIMS | OIDC 기반 SSO를 사용하는 경우 이 변수를 `true`로 설정하여 OIDC 그룹을 기반으로 인스턴스에서 W&B team 멤버십을 자동화합니다. 사용자 OIDC 토큰에 `groups` 클레임을 추가합니다. 각 항목이 사용자가 속해야 하는 W&B team의 이름인 문자열 배열이어야 합니다. 배열에는 사용자가 속한 모든 팀이 포함되어야 합니다. |
| GORILLA_LDAP_GROUP_SYNC | LDAP 기반 SSO를 사용하는 경우 `true`로 설정하여 LDAP 그룹을 기반으로 인스턴스에서 W&B team 멤버십을 자동화합니다. |
| GORILLA_OIDC_CUSTOM_SCOPES | OIDC 기반 SSO를 사용하는 경우 W&B 인스턴스가 ID 공급자로부터 요청해야 하는 추가 [스코프](https://auth0.com/docs/get-started/apis/scopes/openid-connect-scopes)를 지정할 수 있습니다. W&B는 이러한 사용자 지정 스코프로 인해 SSO 기능을 변경하지 않습니다. |
| GORILLA_USE_IDENTIFIER_CLAIMS | OIDC 기반 SSO를 사용하는 경우 이 변수를 `true`로 설정하여 ID 공급자의 특정 OIDC 클레임을 사용하여 사용자의 사용자 이름과 전체 이름을 적용합니다. 설정된 경우 `preferred_username` 및 `name` OIDC 클레임에서 적용된 사용자 이름과 전체 이름을 구성해야 합니다. 사용자 이름에는 영숫자 문자, 밑줄 및 하이픈만 특수 문자로 포함할 수 있습니다. |
| GORILLA_DISABLE_PERSONAL_ENTITY | W&B 인스턴스에서 개인 사용자 project를 끄려면 `true`로 설정하십시오. 설정된 경우 사용자는 개인 entity에서 새 개인 project를 만들 수 없으며 기존 개인 project에 대한 쓰기가 꺼집니다. |
| GORILLA_DISABLE_ADMIN_TEAM_ACCESS | Organization 또는 Instance Admin이 W&B team에 자체적으로 가입하거나 추가하는 것을 제한하려면 `true`로 설정하여 Data & AI persona만 team 내의 project에 엑세스할 수 있도록 합니다. |

{{% alert color="secondary" %}}
W&B는 `GORILLA_DISABLE_ADMIN_TEAM_ACCESS`와 같은 일부 설정을 활성화하기 전에 주의를 기울이고 모든 의미를 이해할 것을 권장합니다. 질문이 있으시면 W&B 팀에 문의하십시오.
{{% /alert %}}
