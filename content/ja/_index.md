---
title: Weights & Biases ドキュメント
---

<div style="padding-top:50px;">&nbsp;</div>
<div style="max-width:1600px; margin: 0 auto">
{{< banner title="Weights & Biases ドキュメント" background="/images/support/support_banner.png" >}}
ドキュメントが必要な製品を選択してください。
{{< /banner >}}

<div class="top-row-cards">
{{< cardpane >}}
{{% card %}}<div onclick="window.location.href='https://weave-docs.wandb.ai'" style="cursor: pointer;">

<div className="card-banner-icon" style="float:left;margin-right:10px !important; margin-top: -12px !important">
<img src="/img/weave-logo.svg" alt="W&B Weave logo" width="50" height="50"/>
</div>
<h2>W&B Weave</h2>

### アプリで AI モデルを活用する

[W&B Weave](https://weave-docs.wandb.ai/) を使用して、コード内の AI モデルを管理します。機能には、トレーシング、出力評価、コスト見積もり、および異なる大規模言語モデル（LLM）と設定を比較するためのホスト型推論サービスとプレイグラウンドが含まれます。

- [イントロダクション](https://weave-docs.wandb.ai/)
- [クイックスタート](https://weave-docs.wandb.ai/quickstart)
- [YouTube デモ](https://www.youtube.com/watch?v=IQcGGNLN3zo)
- [プレイグラウンドを試す](https://weave-docs.wandb.ai/guides/tools/playground/)
- [W&B runs で Weave を使用する]({{< relref path="/guides/weave/set-up-weave" lang="ja" >}})

</div>{{% /card %}}
{{% card %}}<div onclick="window.location.href='/guides'" style="cursor: pointer;">

<div className="card-banner-icon" style="float:left;margin-right:10px !important; margin-top: -12px !important">
<img src="/img/wandb-gold.svg" alt="W&B Models logo" width="40" height="40"/>
</div>
<h2>W&B Models</h2>

### AI モデルを開発する

[W&B Models]({{< relref path="/guides/" lang="ja" >}}) を使用して、AI モデルの開発を管理します。機能には、トレーニング、ファインチューニング、Reports、ハイパーパラメーター探索の自動化、およびバージョン管理と再現性のためのモデルレジストリの活用が含まれます。

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

### ファウンデーション モデルにアクセスする

[W&B Inference]({{< relref path="/guides/inference/" lang="ja" >}}) を使用して、OpenAI 互換 API を介して主要なオープンソースのファウンデーション モデルにアクセスします。機能には、複数のモデルオプション、使用状況追跡、およびトレーシングと評価のための Weave とのインテグレーションが含まれます。

- [イントロダクション]({{< relref path="/guides/inference/" lang="ja" >}})
- [利用可能なモデル]({{< relref path="/guides/inference/models/" lang="ja" >}})
- [API リファレンス]({{< relref path="/guides/inference/api-reference/" lang="ja" >}})
- [プレイグラウンドで試す](https://wandb.ai/inference)

</div>{{% /card %}}
{{< /cardpane >}}
</div>

<div class="bottom-row-cards">
{{< cardpane >}}
{{% card %}}<div onclick="window.location.href='/guides/core/'" style="cursor: pointer; padding-left: 20px">
<h2>コアコンポーネント</h2>

両方の W&B 製品は、AI/ML エンジニアリング作業を可能にし、加速する共通のコンポーネントを共有しています。

- [Registry]({{< relref path="/guides/core/registry/" lang="ja" >}})
- [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}})
- [Reports]({{< relref path="/guides/core/reports/" lang="ja" >}})
- [Automations]({{< relref path="/guides/core/automations/" lang="ja" >}})
- [シークレット]({{< relref path="/guides/core/secrets.md" lang="ja" >}})

</div>{{% /card %}}
{{% card %}}<div onclick="window.location.href='/guides/hosting'" style="cursor: pointer;padding-left:20px;">

<h2>プラットフォーム</h2>

Weights & Biases プラットフォームは、SaaS オファリングを通じてアクセスすることも、オンプレミスにデプロイすることもでき、IAM、セキュリティ、監視、プライバシー機能を提供します。

- [デプロイメントオプション]({{< relref path="/guides/hosting/hosting-options/" lang="ja" >}})
- [ID とアクセス管理 (IAM)]({{< relref path="/guides/hosting/iam/" lang="ja" >}})
- [データセキュリティ]({{< relref path="/guides/hosting/data-security/" lang="ja" >}})
- [プライバシー設定]({{< relref path="/guides/hosting/privacy-settings/" lang="ja" >}})
- [監視と使用状況]({{< relref path="/guides/hosting/monitoring-usage/" lang="ja" >}})

</div>{{% /card %}}
{{% card %}}<div onclick="window.location.href='/support/'" style="cursor: pointer;padding-left:20px;">

<h2>サポート</h2>

Weights & Biases プラットフォームに関するあらゆる側面についてサポートを受けられます。よくある質問、トラブルシューティングガイド、およびサポートチームへの連絡方法を見つけてください。

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