---
title: Does W&B support SSO for Multi-tenant?
menu:
  support:
    identifier: ko-support-kb-articles-sso_multitenant
support:
- security
toc_hide: true
type: docs
url: /ko/support/:filename
translationKey: sso_multitenant
---
W&B는 Auth0을 통해 멀티 테넌트 제품에 대한 Single Sign-On (SSO)을 지원합니다. SSO 인테그레이션은 Okta 또는 Azure AD와 같은 모든 OIDC 호환 ID 공급자와 호환됩니다. OIDC 공급자를 구성하려면 다음 단계를 따르세요.

* ID 공급자에서 Single Page Application (SPA)을 만듭니다.
* `grant_type`을 `implicit` 플로우로 설정합니다.
* 콜백 URI를 `https://wandb.auth0.com/login/callback`으로 설정합니다.

**W&B 요구 사항**

설정이 완료되면 애플리케이션의 `Client ID` 및 `Issuer URL`을 고객 성공 관리자 (CSM)에게 문의하세요. W&B는 이러한 세부 정보를 사용하여 Auth0 연결을 설정하고 SSO를 활성화합니다.
