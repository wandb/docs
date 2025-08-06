---
title: W&B 는 멀티 테넌트 환경에서 SSO(싱글 사인온)를 지원하나요?
menu:
  support:
    identifier: ko-support-kb-articles-sso_multitenant
support:
- 보안
toc_hide: true
type: docs
url: /support/:filename
---

W&B 는 Auth0 를 통해 멀티테넌트 환경에서 Single Sign-On (SSO) 을 지원합니다. SSO 인테그레이션은 Okta, Azure AD 와 같은 OIDC 호환 아이덴티티 제공자와도 호환됩니다. OIDC 제공자를 설정하려면 다음 단계를 따르세요:

* 아이덴티티 제공자에서 Single Page Application (SPA) 을 생성합니다.
* `grant_type` 을 `implicit` 플로우로 설정합니다.
* 콜백 URI 를 `https://wandb.auth0.com/login/callback` 으로 설정합니다.

**W&B 를 위한 요구 사항**

설정이 완료되면, 애플리케이션의 `Client ID` 와 `Issuer URL` 을 고객 성공 매니저 (CSM) 에게 전달해 주세요. W&B 팀이 이 정보를 기반으로 Auth0 연결을 설정하고 SSO 를 활성화합니다.