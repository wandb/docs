---
title: Advanced IAM configuration
displayed_sidebar: default
---

기본 [환경 변수](../env-vars.md) 외에도, 환경 변수를 사용하여 [전용 클라우드](../hosting-options/dedicated_cloud.md) 또는 [자가 관리](../hosting-options/self-managed.md) 인스턴스에 대한 IAM 옵션을 구성할 수 있습니다.

IAM 요구 사항에 따라 인스턴스를 위해 다음 환경 변수 중 하나를 선택하세요.

| Environment variable | Description |
|----------------------|-------------|
| DISABLE_SSO_PROVISIONING | W&B 인스턴스에서 사용자 자동 프로비저닝을 비활성화하려면 이를 `true`로 설정하세요. |
| SESSION_LENGTH | 기본 사용자 세션 만료 시간을 변경하려면 이 변수를 원하는 시간 수로 설정하세요. 예를 들어, 세션 만료 시간을 24시간으로 설정하려면 SESSION_LENGTH를 `24`로 설정하세요. 기본 값은 720시간입니다. |
| GORILLA_ENABLE_SSO_GROUP_CLAIMS | OIDC 기반 SSO를 사용하는 경우, OIDC 그룹에 기반하여 인스턴스에서 W&B 팀 멤버십을 자동화하려면 이 변수를 `true`로 설정하세요. 사용자 OIDC 토큰에 `groups` 클레임을 추가하세요. 각 항목이 사용자가 속해야 하는 W&B 팀 이름인 문자열 배열이어야 합니다. 배열에는 사용자가 속한 모든 팀이 포함되어야 합니다. |
| GORILLA_LDAP_GROUP_SYNC | LDAP 기반 SSO를 사용하는 경우, LDAP 그룹에 기반하여 인스턴스에서 W&B 팀 멤버십을 자동화하려면 이를 `true`로 설정하세요. |
| GORILLA_OIDC_CUSTOM_SCOPES | OIDC 기반 SSO를 사용하는 경우, W&B 인스턴스가 식별 제공자에게 요청해야 하는 추가 [스코프](https://auth0.com/docs/get-started/apis/scopes/openid-connect-scopes)를 지정할 수 있습니다. W&B는 이러한 사용자 정의 스코프로 인해 SSO 기능을 변경하지 않습니다. |
| GORILLA_USE_IDENTIFIER_CLAIMS | OIDC 기반 SSO를 사용하는 경우, 식별 제공자의 특정 OIDC 클레임을 사용하여 사용자의 사용자 이름과 전체 이름을 적용하려면 이 변수를 `true`로 설정하세요. 설정된 경우, `preferred_username`과 `name` OIDC 클레임에 적용된 사용자 이름과 전체 이름을 반드시 구성하세요. 사용자 이름은 밑줄과 하이픈을 특수 문자로 사용하여 영숫자만 포함할 수 있습니다. |
| GORILLA_DISABLE_PERSONAL_ENTITY | W&B 인스턴스에서 개인 사용자 프로젝트를 비활성화하려면 이를 `true`로 설정하세요. 설정되면, 사용자는 자신의 개인 엔터티에서 새로운 개인 프로젝트를 생성할 수 없으며, 기존 개인 프로젝트에 대한 쓰기도 비활성화됩니다. |
| GORILLA_DISABLE_ADMIN_TEAM_ACCESS | 조직 또는 인스턴스 관리자가 W&B 팀에 자신을 추가하거나 스스로 참여하는 것을 제한하려면 이를 `true`로 설정하세요. 이는 Data & AI 인물만이 팀 내의 프로젝트에 접근할 수 있도록 보장합니다. |

:::caution
W&B는 `GORILLA_DISABLE_ADMIN_TEAM_ACCESS`와 같은 일부 설정을 활성화하기 전에 주의하고 모든 영향을 이해할 것을 권장합니다. 질문이 있으면 W&B 팀에 문의하세요.
:::