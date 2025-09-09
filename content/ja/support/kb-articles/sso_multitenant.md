---
title: W&B は マルチテナント向けの SSO をサポートしていますか？
menu:
  support:
    identifier: ja-support-kb-articles-sso_multitenant
support:
- セキュリティ
toc_hide: true
type: docs
url: /support/:filename
---

W&B は Auth0 を通じてマルチテナント版でのシングルサインオン (SSO) に対応しています。SSO インテグレーションは Okta や Azure AD など、あらゆる OIDC 準拠のアイデンティティプロバイダと互換性があります。OIDC プロバイダを設定するには、次の手順に従ってください:

* アイデンティティプロバイダ上で シングルページ アプリケーション (SPA) を作成します。
* `grant_type` を `implicit` フローに設定します。
* コールバック URI を `https://wandb.auth0.com/login/callback` に設定します。

**W&B の要件**

セットアップが完了したら、アプリケーションの `Client ID` と `Issuer URL` を添えて、カスタマーサクセスマネージャー (CSM) に連絡してください。W&B はこれらの情報を用いて Auth0 接続を確立し、SSO を有効化します。