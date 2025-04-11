---
title: W&B は マルチテナント に SSO をサポートしていますか？
menu:
  support:
    identifier: ja-support-kb-articles-sso_multitenant
support:
- security
toc_hide: true
type: docs
url: /support/:filename
---

W&B は、Auth0 を通じたマルチテナント提供のためにシングルサインオン (SSO) をサポートしています。SSO インテグレーションは、Okta や Azure AD など、OIDC 準拠の任意のアイデンティティプロバイダーと互換性があります。OIDC プロバイダーを設定するには、次の手順に従います:

* アイデンティティプロバイダーでシングルページアプリケーション (SPA) を作成します。
* `grant_type` を `implicit` フローに設定します。
* コールバック URI を `https://wandb.auth0.com/login/callback` に設定します。

**W&B の要件**

セットアップを完了したら、アプリケーションの `Client ID` と `Issuer URL` をお客様サクセスマネージャー (CSM) に連絡してください。W&B はこれらの詳細を使用して Auth0 接続を確立し、SSO を有効にします。