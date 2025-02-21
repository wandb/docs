---
title: Support
cascade:
- url: support/:filename
menu:
  support:
    identifier: ja-support-_index
    parent: null
no_list: true
type: docs
url: support
---

{{< banner title="お手伝いできることはありますか？" background="/images/support/support_banner.png" >}}
サポート記事、製品ドキュメント、<br>
および W&B コミュニティからヘルプを検索。
{{< /banner >}}

## 注目の記事

すべてのカテゴリで最もよくある質問はこちらです。

* [`wandb.init` はトレーニング プロセスに何をしますか？]({{< relref path="./wandbinit_training_process.md" lang="ja" >}})
* [カスタム CLI コマンドを Sweeps で使用するにはどうすればよいですか？]({{< relref path="./custom_cli_commands_sweeps.md" lang="ja" >}})
* [メトリクスをオフラインで保存し、後で W&B に同期することは可能ですか？]({{< relref path="./same_metric_appearing_more.md" lang="ja" >}})
* [トレーニング コードで run の名前を設定するにはどうすればよいですか？]({{< relref path="./configure_name_run_training_code.md" lang="ja" >}})

お探しのものが見つからない場合は、[人気カテゴリ]({{< relref path="#popular-categories" lang="ja" >}})を以下で閲覧するか、カテゴリに基づいて記事を検索してください。

## 人気カテゴリ

カテゴリ別に記事を閲覧。

{{< cardpane >}}
  {{< card >}}
    <a href="index_experiments">
      <h2 className="card-title">Experiments</h2>
      <p className="card-content">機械学習実験を追跡、視覚化、および比較</p>
    </a>
  {{< /card >}}
  {{< card >}}
    <a href="index_artifacts">
      <h2 className="card-title">Artifacts</h2>
      <p className="card-content">データセット、モデル、およびその他の機械学習アーティファクトをバージョン管理して追跡</p>
    </a>
  {{< /card >}}
{{< /cardpane >}}
{{< cardpane >}}
  {{< card >}}
    <a href="index_reports">
      <h2 className="card-title">Reports</h2>
      <p className="card-content">作業を共有するためのインタラクティブで共同作業可能なレポートを作成</p>
    </a>
  {{< /card >}}
  {{< card >}}
    <a href="index_sweeps">
      <h2 className="card-title">Sweeps</h2>
      <p className="card-content">ハイパーパラメータ探索を自動化</p>
    </a>
  {{< /card >}}
{{< /cardpane >}}

{{< card >}}
  <div className="card-banner-icon" style="float:left;margin-right:10px !important; margin-top: -12px !important">
    {{< img src="/images/support/callout-icon.svg" alt="Callout Icon" width="32" height="32" >}}
  </div>
  <h2>まだお探しのものが見つからない場合は？</h2>
  <a href="mailto:support@wandb.com" className="contact-us-button">
    サポートに連絡する
  </a>
{{< /card >}}