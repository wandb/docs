---
title: W&B アプリ UI
menu:
  default:
    identifier: w-b-app-ui-reference
    parent: models
url: guides/app
aliases:
- /guides/models/app/features
---

このセクションでは、W&B アプリ UI の利用方法について詳しく説明します。Workspace、Teams、Registry の管理、実験の可視化と観察、パネルや Reports の作成、Automations の設定など、多彩な機能をご活用いただけます。

W&B アプリにはウェブブラウザでアクセスできます。

- W&B マルチテナントデプロイメントは、公開ウェブ上（https://wandb.ai/）からアクセスできます。
- W&B 専用クラウドデプロイメントは、W&B Dedicated Cloud ご契約時に設定したドメインでアクセスできます。管理者ユーザーは W&B Management Console でドメインを更新できます。右上のアイコンをクリックし、**System console** を選択してください。
- W&B セルフマネージドデプロイメントは、デプロイ時に設定したホスト名でアクセスできます。たとえば、Helm でデプロイする場合、ホスト名は `values.global.host` で設定されます。管理者ユーザーは W&B Management Console でドメインを更新できます。右上のアイコンをクリックし、**System console** を選択してください。

さらに詳しく知りたい方へ：

- [実験をトラッキングする]({{< relref "/guides/models/track/" >}})：Runs や Sweeps を活用。
- [デプロイメント設定を変更する]({{< relref "settings-page/" >}})・[デフォルト値を設定する]({{< relref "features/cascade-settings.md" >}})。
- [パネルを追加して実験を可視化する]({{< relref "/guides/models/app/features/panels/" >}})：折れ線グラフ、棒グラフ、メディアパネル、クエリパネル、Tables など。
- [カスタムチャートを追加する]({{< relref "/guides/models/app/features/custom-charts/" >}})。
- [Reports を作成・共有する]({{< relref "/guides/core/reports/" >}})。