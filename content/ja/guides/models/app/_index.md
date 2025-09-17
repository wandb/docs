---
title: W&B アプリの UI
aliases:
- /guides/models/app/features
menu:
  default:
    identifier: ja-guides-models-app-_index
    parent: models
url: guides/app
---

このセクションでは、W&B App の UI の使い方に役立つ詳細を紹介します。Workspaces や Teams、レジストリの管理、Experiments の可視化と観察、パネルや Reports の作成、オートメーションの設定などが行えます。

W&B App には Web ブラウザから アクセス できます。

- W&B の Multi-tenant デプロイメントは、パブリック Web 上の https://wandb.ai/ から アクセス できます。
- W&B 専用クラウド デプロイメントは、W&B Dedicated Cloud にサインアップした際に設定したドメインで アクセス できます。ドメインは管理者 ユーザー が W&B Management Console で更新できます。右上のアイコンをクリックし、**System console** をクリックします。
- W&B Self-Managed デプロイメントは、W&B をデプロイした際に設定したホスト名で アクセス できます。たとえば Helm でデプロイする場合、ホスト名は `values.global.host` で設定します。ドメインは管理者 ユーザー が W&B Management Console で更新できます。右上のアイコンをクリックし、**System console** をクリックします。

詳しくは次を参照してください:

- [Experiments をトラッキング]({{< relref path="/guides/models/track/" lang="ja" >}})。Runs または Sweeps を使用します。
- [デプロイメントの 設定 を構成]({{< relref path="settings-page/" lang="ja" >}}) と [デフォルト]({{< relref path="features/cascade-settings.md" lang="ja" >}})。
- [パネルを追加]({{< relref path="/guides/models/app/features/panels/" lang="ja" >}})して Experiments を可視化します。例: 折れ線プロット、棒グラフ、メディア パネル、クエリ パネル、Tables。
- [カスタム チャートを追加]({{< relref path="/guides/models/app/features/custom-charts/" lang="ja" >}})。
- [Reports を作成して共有]({{< relref path="/guides/core/reports/" lang="ja" >}})。