---
title: W&B はマルチテナント環境で SSO をサポートしていますか？
url: /support/:filename
toc_hide: true
type: docs
support:
- セキュリティ
---

W&B は、Auth0 を通じたマルチテナント向けのシングルサインオン (SSO) をサポートしています。SSO インテグレーションは、Okta や Azure AD など、OIDC 準拠のアイデンティティプロバイダーで互換性があります。OIDC プロバイダーを設定するには、以下の手順に従ってください。

* アイデンティティプロバイダー上で Single Page Application (SPA) を作成します。
* `grant_type` を `implicit` フローに設定します。
* コールバック URI を `https://wandb.auth0.com/login/callback` に設定します。

**W&B の要件**

セットアップが完了したら、アプリケーションの `Client ID` と `Issuer URL` をカスタマーサクセスマネージャー (CSM) にご連絡ください。W&B がこれらの情報を利用して Auth0 接続を確立し、SSO を有効にします。