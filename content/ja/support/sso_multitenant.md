---
title: Does W&B support SSO for Multi-tenant?
menu:
  support:
    identifier: ja-support-sso_multitenant
tags:
- security
toc_hide: true
type: docs
---

W&B は、Auth0 を介してマルチテナント製品のシングルサインオン (SSO) をサポートしています。SSO の インテグレーション は、Okta や Azure AD などの OIDC 準拠の ID プロバイダーと互換性があります。OIDC プロバイダーを構成するには、次の手順に従ってください。

* ID プロバイダーでシングルページ アプリケーション (SPA) を作成します。
* `grant_type` を `implicit` フローに設定します。
* コールバック URI を `https://wandb.auth0.com/login/callback` に設定します。

**W&B の要件**

設定が完了したら、アプリケーションの `Client ID` と `Issuer URL` をカスタマーサクセスマネージャー (CSM) に連絡してください。W&B は、これらの詳細を使用して Auth0 接続を確立し、SSO を有効にします。
