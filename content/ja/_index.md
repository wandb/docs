---
title: Weights & Biases ドキュメント
---

<div style="padding-top:50px;">&nbsp;</div>
<div style="max-width:1600px; margin: 0 auto">
{{< banner title="Weights & Biases ドキュメント" background="/images/support/support_banner.png" >}}
ご利用になりたいプロダクトのドキュメントをお選びください。
{{< /banner >}}

<div class="top-row-cards">
{{< cardpane >}}
{{% card %}}<div onclick="window.location.href='https://weave-docs.wandb.ai'" style="cursor: pointer;">

<div className="card-banner-icon" style="float:left;margin-right:10px !important; margin-top: -12px !important">
<img src="/img/weave-logo.svg" alt="W&B Weave logo" width="50" height="50"/>
</div>
<h2>W&B Weave</h2>

### AIモデルをアプリで活用

[W&B Weave](https://weave-docs.wandb.ai/) を使うと、コード内でのAIモデル管理が可能です。トレーシング、出力評価、コスト見積もり、さまざまな大規模言語モデル（LLM）や設定を比較できる推論サービスやプレイグラウンドなどの機能を備えています。

- [イントロダクション](https://weave-docs.wandb.ai/)
- [クイックスタート](https://weave-docs.wandb.ai/quickstart)
- [YouTube デモ](https://www.youtube.com/watch?v=IQcGGNLN3zo)
- [プレイグラウンドを試す](https://weave-docs.wandb.ai/guides/tools/playground/)

</div>{{% /card %}}
{{% card %}}<div onclick="window.location.href='/guides'" style="cursor: pointer;">

<div className="card-banner-icon" style="float:left;margin-right:10px !important; margin-top: -12px !important">
<img src="/img/wandb-gold.svg" alt="W&B Models logo" width="40" height="40"/>
</div>
<h2>W&B Models</h2>

### AIモデル開発

[W&B Models]({{< relref path="/guides/" lang="ja" >}}) を活用してAIモデル開発を管理できます。トレーニング、ファインチューニング、レポーティング、ハイパーパラメーター探索の自動化、モデルレジストリによるバージョン管理・再現性などの機能を備えています。

- [イントロダクション]({{< relref path="/guides/" lang="ja" >}})
- [クイックスタート]({{< relref path="/guides/quickstart/" lang="ja" >}})
- [YouTube チュートリアル](https://www.youtube.com/watch?v=tHAFujRhZLA)
- [オンラインコース](https://wandb.ai/site/courses/101/)

</div>{{% /card %}}
{{% card %}}<div onclick="window.location.href='/guides/inference/'" style="cursor: pointer;">

<div className="card-banner-icon" style="float:left;margin-right:10px !important; margin-top: -12px !important">
<img src="/img/wandb-gold.svg" alt="W&B Inference logo" width="40" height="40"/>
</div>
<h2>W&B Inference</h2>

### ファウンデーションモデルへのアクセス

[W&B Inference]({{< relref path="/guides/inference/" lang="ja" >}}) を使って、OpenAI 互換API経由で最新のオープンソースファウンデーションモデルにアクセスできます。複数モデルからの選択、利用状況のトラッキング、Weave との連携によるトレーシングや評価が可能です。

- [イントロダクション]({{< relref path="/guides/inference/" lang="ja" >}})
- [利用可能なモデル一覧]({{< relref path="/guides/inference/models/" lang="ja" >}})
- [APIリファレンス]({{< relref path="/guides/inference/api-reference/" lang="ja" >}})
- [プレイグラウンドで試す](https://wandb.ai/inference)

</div>{{% /card %}}
{{< /cardpane >}}
</div>

<div class="bottom-row-cards">
{{< cardpane >}}
{{% card %}}<div onclick="window.location.href='/guides/core/'" style="cursor: pointer; padding-left: 20px">
<h2>コアコンポーネント</h2>

W&B の各プロダクトに共通する、AI/ML エンジニアリングを支える基盤機能です。

- [Registry]({{< relref path="/guides/core/registry/" lang="ja" >}})
- [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}})
- [Reports]({{< relref path="/guides/core/reports/" lang="ja" >}})
- [Automations]({{< relref path="/guides/core/automations/" lang="ja" >}})
- [Secrets]({{< relref path="/guides/core/secrets.md" lang="ja" >}})

</div>{{% /card %}}
{{% card %}}<div onclick="window.location.href='/guides/hosting'" style="cursor: pointer;padding-left:20px;">

<h2>プラットフォーム</h2>

Weights & Biases プラットフォームは SaaS 提供またはオンプレミス展開が可能で、IAM、セキュリティ、モニタリング、プライバシー保護の各種機能を提供します。

- [デプロイメントオプション]({{< relref path="/guides/hosting/hosting-options/" lang="ja" >}})
- [ID・アクセス管理（IAM）]({{< relref path="/guides/hosting/iam/" lang="ja" >}})
- [データセキュリティ]({{< relref path="/guides/hosting/data-security/" lang="ja" >}})
- [プライバシー設定]({{< relref path="/guides/hosting/privacy-settings/" lang="ja" >}})
- [モニタリングと利用状況]({{< relref path="/guides/hosting/monitoring-usage/" lang="ja" >}})

</div>{{% /card %}}
{{% card %}}<div onclick="window.location.href='/support/'" style="cursor: pointer;padding-left:20px;">

<h2>サポート</h2>

Weights & Biases プラットフォーム全般に関するサポートを提供しています。よくある質問への回答、トラブルシューティングガイド、サポートチームへのお問い合わせ方法も掲載。

- [ナレッジベース記事]({{< relref path="/support/" lang="ja" >}})
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