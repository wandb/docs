---
title: 고급 IAM 설정
menu:
  default:
    identifier: ko-guides-hosting-iam-advanced_env_vars
    parent: identity-and-access-management-iam
---

기본 [환경 변수]({{< relref path="../env-vars.md" lang="ko" >}}) 외에도, [Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud.md" lang="ko" >}}) 또는 [Self-managed]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}}) 인스턴스의 IAM 옵션을 구성하기 위해 환경 변수를 사용할 수 있습니다.

IAM 요구사항에 따라 아래 환경 변수 중에서 인스턴스에 맞는 것을 선택해 설정하세요.

| 환경 변수 | 설명 |
|----------------------|-------------|
| `DISABLE_SSO_PROVISIONING` | 이 값을 `true`로 설정하면 W&B 인스턴스에서 사용자 자동 프로비저닝이 비활성화됩니다. |
| `SESSION_LENGTH` | 기본 사용자 세션 만료 시간을 변경하려면, 이 변수를 원하는 시간(시간 단위)으로 설정하세요. 예를 들어, SESSION_LENGTH를 `24`로 설정하면 세션 만료 시간이 24시간으로 변경됩니다. 기본 값은 720시간입니다. |
| `GORILLA_ENABLE_SSO_GROUP_CLAIMS` | OIDC 기반 SSO를 사용하는 경우, 이 변수를 `true`로 설정하면 OIDC 그룹에 따라 W&B 팀 멤버십이 자동으로 부여됩니다. 사용자 OIDC 토큰에 `groups` 클레임을 추가해야 하며, 각 항목이 사용자가 소속되어야 하는 W&B 팀의 이름인 문자열 배열이어야 합니다. 배열에는 사용자가 속한 모든 팀이 포함되어야 합니다. |
| `GORILLA_LDAP_GROUP_SYNC` | LDAP 기반 SSO를 사용하는 경우, 이 값을 `true`로 설정하면 LDAP 그룹에 따라 W&B 팀 멤버십이 자동으로 관리됩니다. |
| `GORILLA_OIDC_CUSTOM_SCOPES` | OIDC 기반 SSO를 사용하는 경우, [scope](https://auth0.com/docs/get-started/apis/scopes/openid-connect-scopes)를 추가로 지정하여 W&B 인스턴스가 아이덴티티 프로바이더에 요청하는 권한을 확장할 수 있습니다. 이 커스텀 scope들은 SSO 기능에는 아무런 영향도 주지 않습니다. |
| `GORILLA_USE_IDENTIFIER_CLAIMS` | OIDC 기반 SSO를 사용하는 경우, 이 변수를 `true`로 설정하면 아이덴티티 프로바이더의 특정 OIDC 클레임을 사용해 사용자 이름과 전체 이름이 강제 적용됩니다. 설정 시 `preferred_username` 및 `name` OIDC 클레임에 각각 사용자 이름과 전체 이름이 들어가야 합니다. 사용자 이름에는 영숫자, 밑줄, 하이픈만 사용할 수 있습니다. |
| `GORILLA_DISABLE_PERSONAL_ENTITY` | true로 설정하면 [personal entities]({{< relref path="/support/kb-articles/difference_team_entity_user_entity_mean_me.md" lang="ko" >}})가 비활성화됩니다. 새로운 personal projects 생성 및 기존 personal projects에 쓰기를 막을 수 있습니다. |
| `GORILLA_DISABLE_ADMIN_TEAM_ACCESS` | 이 값을 `true`로 설정하면 조직 또는 인스턴스 Admin이 스스로 W&B 팀에 가입하거나 본인을 추가하는 것을 제한합니다. 이를 통해 팀 내 프로젝트에 데이터 및 AI 담당자만 엑세스하도록 할 수 있습니다. |
| `WANDB_IDENTITY_TOKEN_FILE`        | [Identity federation]({{< relref path="/guides/hosting/iam/authentication/identity_federation.md" lang="ko" >}})의 경우, JWT(Java Web Token)가 저장되는 로컬 디렉토리의 절대 경로를 지정합니다. |

{{% alert color="secondary" %}}
일부 설정(`GORILLA_DISABLE_ADMIN_TEAM_ACCESS` 등)을 활성화하기 전에 반드시 유의 사항을 숙지하고 신중하게 결정하는 것이 좋습니다. 궁금한 점이 있으면 언제든 W&B 팀에 문의하세요.
{{% /alert %}}