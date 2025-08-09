---
title: W&B アプリ UI
aliases:
- /guides/models/app/features
menu:
  default:
    identifier: ja-guides-models-app-_index
    parent: models
url: guides/app
---

このセクションでは、W&B App の UI を活用するための詳細情報を提供します。Workspace、Teams、レジストリの管理、Experiments の可視化および観察、パネルや Reports の作成、Automations の設定などが行えます。

W&B App にはウェブブラウザーからアクセスできます。

- W&B Multi-tenant デプロイメントはパブリックウェブ上の https://wandb.ai/ でアクセス可能です。
- W&B Dedicated Cloud デプロイメントには、W&B Dedicated Cloud にサインアップした際に設定したドメインでアクセスできます。管理者ユーザーは W&B Management Console でドメインを更新できます。右上のアイコンをクリックし、**System console** をクリックしてください。
- W&B Self-Managed デプロイメントは、デプロイ時に設定したホスト名からアクセスできます。たとえば Helm を使ってデプロイした場合、ホスト名は `values.global.host` で設定されています。管理者ユーザーは W&B Management Console でドメインを更新できます。右上のアイコンをクリックし、**System console** をクリックしてください。

詳しくはこちら：

- [Experiments をトラッキングする]({{< relref path="/guides/models/track/" lang="ja" >}})（run や sweep を利用）
- [デプロイメント設定を行う]({{< relref path="settings-page/" lang="ja" >}})、および [デフォルト値の設定]({{< relref path="features/cascade-settings.md" lang="ja" >}})
- [パネルを追加して]({{< relref path="/guides/models/app/features/panels/" lang="ja" >}})、Experiments を可視化（ラインプロット、バープロット、メディアパネル、クエリパネル、テーブルなど）
- [カスタムチャートの追加]({{< relref path="/guides/models/app/features/custom-charts/" lang="ja" >}})
- [Reports の作成と共有]({{< relref path="/guides/core/reports/" lang="ja" >}})