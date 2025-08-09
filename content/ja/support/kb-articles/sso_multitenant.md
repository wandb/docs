---
title: W&Bはマルチテナント向けのSSOをサポートしていますか？
menu:
  support:
    identifier: ja-support-kb-articles-sso_multitenant
support:
- セキュリティ
toc_hide: true
type: docs
url: /support/:filename
---

W&B では、Auth0 を通じたマルチテナント向けのシングルサインオン（SSO）をサポートしています。SSO インテグレーションは、Okta や Azure AD など、OIDC に準拠した任意のアイデンティティプロバイダーと互換性があります。OIDC プロバイダーを設定するには、以下の手順を実行してください。

* アイデンティティプロバイダー上で Single Page Application (SPA) を作成します。
* `grant_type` を `implicit` フローに設定します。
* コールバック URI を `https://wandb.auth0.com/login/callback` に設定します。

**W&B への要件**

セットアップ完了後、アプリケーションの `Client ID` と `Issuer URL` をカスタマーサクセスマネージャー（CSM）へご連絡ください。W&B 側でこれらの情報を使って Auth0 コネクションを作成し、SSO を有効化します。