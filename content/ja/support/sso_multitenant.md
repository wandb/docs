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

W&B は、Multi-tenant 提供のために Auth0 を通じてシングルサインオン (SSO) をサポートしています。SSO インテグレーションは、Okta や Azure AD など、OIDC に準拠した任意のアイデンティティプロバイダと互換性があります。OIDC プロバイダを設定するには、以下の手順に従ってください:

* アイデンティティプロバイダ上でシングルページアプリケーション (SPA) を作成します。
* `grant_type` を `implicit` フローに設定します。
* コールバック URI を `https://wandb.auth0.com/login/callback` に設定します。

**W&B の要件**

セットアップを完了したら、アプリケーションの `Client ID` と `Issuer URL` を持ってカスタマーサクセスマネージャー (CSM) に連絡してください。W&B はこれらの詳細を使用して Auth0 接続を確立し、SSO を有効にします。