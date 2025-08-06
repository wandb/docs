---
title: Weights & Biases ドキュメント
---

<div style="padding-top:50px;">&nbsp;</div>
<div style="max-width:1600px; margin: 0 auto">
{{< banner title="Weights & Biases ドキュメント" background="/images/support/support_banner.png" >}}
必要なプロダクトのドキュメントを選択してください。
{{< /banner >}}

<div class="top-row-cards">
{{< cardpane >}}
{{% card %}}<div onclick="window.location.href='https://weave-docs.wandb.ai'" style="cursor: pointer;">

<div className="card-banner-icon" style="float:left;margin-right:10px !important; margin-top: -12px !important">
<img src="/img/weave-logo.svg" alt="W&B Weave logo" width="50" height="50"/>
</div>
<h2>W&B Weave</h2>

### AIモデルをあなたのアプリに活用

[W&B Weave](https://weave-docs.wandb.ai/) を使って、コード内で AI モデルを管理できます。トレース、出力評価、コスト見積もりのほか、複数の大規模言語モデル（LLM）や設定の比較ができるホスティング済みの推論サービス＆プレイグラウンドなどの機能を提供しています。

- [イントロダクション](https://weave-docs.wandb.ai/)
- [クイックスタート](https://weave-docs.wandb.ai/quickstart)
- [YouTube デモ](https://www.youtube.com/watch?v=IQcGGNLN3zo)
- [プレイグラウンドで試す](https://weave-docs.wandb.ai/guides/tools/playground/)

</div>{{% /card %}}
{{% card %}}<div onclick="window.location.href='/guides'" style="cursor: pointer;">

<div className="card-banner-icon" style="float:left;margin-right:10px !important; margin-top: -12px !important">
<img src="/img/wandb-gold.svg" alt="W&B Models logo" width="40" height="40"/>
</div>
<h2>W&B Models</h2>

### AIモデルを開発する

[W&B Models]({{< relref "/guides/" >}}) を使って AI モデル開発を管理しましょう。トレーニング、ファインチューニング、レポーティング、ハイパーパラメーター探索の自動化、モデルレジストリによるバージョン管理や再現性の確保、といった機能が利用できます。

- [イントロダクション]({{< relref "/guides/" >}})
- [クイックスタート]({{< relref "/guides/quickstart/" >}})
- [YouTube チュートリアル](https://www.youtube.com/watch?v=tHAFujRhZLA)
- [オンラインコース](https://wandb.ai/site/courses/101/)

</div>{{% /card %}}
{{% card %}}<div onclick="window.location.href='/guides/inference/'" style="cursor: pointer;">

<div className="card-banner-icon" style="float:left;margin-right:10px !important; margin-top: -12px !important">
<img src="/img/wandb-gold.svg" alt="W&B Inference logo" width="40" height="40"/>
</div>
<h2>W&B Inference</h2>

### ファウンデーションモデルへアクセス

[W&B Inference]({{< relref "/guides/inference/" >}}) を使えば、有力なオープンソースのファウンデーションモデルを OpenAI 互換 API 経由で活用可能です。複数モデルの選択、利用状況のトラッキングに加え、Weave との連携でトレースや評価も簡単です。

- [イントロダクション]({{< relref "/guides/inference/" >}})
- [利用可能モデル]({{< relref "/guides/inference/models/" >}})
- [API リファレンス]({{< relref "/guides/inference/api-reference/" >}})
- [プレイグラウンドで試す](https://wandb.ai/inference)

</div>{{% /card %}}
{{< /cardpane >}}
</div>

<div class="bottom-row-cards">
{{< cardpane >}}
{{% card %}}<div onclick="window.location.href='/guides/core/'" style="cursor: pointer; padding-left: 20px">
<h2>コアコンポーネント</h2>

W&B の各プロダクトは、AI/ML エンジニアリングを支える共通コンポーネントを備えています。

- [Registry]({{< relref "/guides/core/registry/" >}})
- [Artifacts]({{< relref "/guides/core/artifacts/" >}})
- [Reports]({{< relref "/guides/core/reports/" >}})
- [Automations]({{< relref "/guides/core/automations/" >}})
- [Secrets]({{< relref "/guides/core/secrets.md" >}})

</div>{{% /card %}}
{{% card %}}<div onclick="window.location.href='/guides/hosting'" style="cursor: pointer;padding-left:20px;">

<h2>プラットフォーム</h2>

Weights & Biases のプラットフォームは SaaS での利用も、オンプレミス展開も可能です。IAM、セキュリティ、モニタリング、プライバシー機能を提供します。

- [デプロイメントオプション]({{< relref "/guides/hosting/hosting-options/" >}})
- [アイデンティティとアクセス管理 (IAM)]({{< relref "/guides/hosting/iam/" >}})
- [データセキュリティ]({{< relref "/guides/hosting/data-security/" >}})
- [プライバシー設定]({{< relref "/guides/hosting/privacy-settings/" >}})
- [モニタリングと利用状況]({{< relref "/guides/hosting/monitoring-usage/" >}})

</div>{{% /card %}}
{{% card %}}<div onclick="window.location.href='/support/'" style="cursor: pointer;padding-left:20px;">

<h2>サポート</h2>

Weights & Biases プラットフォーム全般に関するご質問やお問い合わせに対応します。よくある質問、トラブルシュートガイド、サポートチームへの連絡方法なども掲載しています。

- [ナレッジベース記事]({{< relref "/support/" >}})
- [コミュニティフォーラム](https://wandb.ai/community)
- [Discord サーバー](https://discord.com/invite/RgB8CPk2ce)
- [サポートへのお問い合わせ](https://wandb.ai/site/contact/)

</div>{{% /card %}}
{{< /cardpane >}}
</div>




</div>


<style>
.td-card-group { margin: 0 auto }
p { overflow: hidden; display: block; }
ul { margin-left: 50px; }

/* Make all cards uniform size in 3x2 grid */
.top-row-cards .td-card-group,
.bottom-row-cards .td-card-group {
    max-width: 100%;
    display: flex;
    justify-content: center;
}

.td-card {
    max-width: 480px !important;
    min-width: 480px !important;
    margin: 0.75rem !important;
    flex: 0 0 auto;
}

/* Ensure consistent height for all cards */
.td-card .card {
    height: 100%;
    min-height: 320px;
}
</style>