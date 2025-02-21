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

{{< banner title="お困りですか？" background="/images/support/support_banner.png" >}}
サポート記事、製品ドキュメント、<br>
W&B コミュニティからヘルプを検索してください。
{{< /banner >}}

## 注目の記事

すべてのカテゴリで最もよくある質問を紹介します。

* [ `wandb.init` はトレーニング プロセスに何をするのですか？]({{< relref path="./wandbinit_training_process.md" lang="ja" >}})
* [ Sweeps でカスタム CLI コマンドを使用するにはどうすればよいですか？]({{< relref path="./custom_cli_commands_sweeps.md" lang="ja" >}})
* [メトリクスをオフラインで保存し、後で W&B に同期することはできますか？]({{< relref path="./same_metric_appearing_more.md" lang="ja" >}})
* [トレーニング コードで run の名前を設定するにはどうすればよいですか？]({{< relref path="./configure_name_run_training_code.md" lang="ja" >}})

お探しのものが見つからない場合は、以下の[人気のカテゴリ]({{< relref path="#popular-categories" lang="ja" >}})を参照するか、カテゴリに基づいて記事を検索してください。

## 人気のカテゴリ

カテゴリ別に記事を閲覧します。

{{< cardpane >}}
  {{< card >}}
    <a href="index_experiments">
      <h2 className="card-title">Experiments</h2>
      <p className="card-content">機械学習 の Experiments を追跡、視覚化、比較します</p>
    </a>
  {{< /card >}}
  {{< card >}}
    <a href="index_artifacts">
      <h2 className="card-title">Artifacts</h2>
      <p className="card-content">データセット、モデル、その他の機械学習 Artifacts をバージョン管理および追跡します</p>
    </a>
  {{< /card >}}
{{< /cardpane >}}
{{< cardpane >}}
  {{< card >}}
    <a href="index_reports">
      <h2 className="card-title">Reports</h2>
      <p className="card-content">インタラクティブな共同 Reports を作成して、作業を共有します</p>
    </a>
  {{< /card >}}
  {{< card >}}
    <a href="index_sweeps">
      <h2 className="card-title">Sweeps</h2>
      <p className="card-content">ハイパーパラメーター の探索を自動化します</p>
    </a>
  {{< /card >}}
{{< /cardpane >}}

{{< card >}}
  <div className="card-banner-icon" style="float:left;margin-right:10px !important; margin-top: -12px !important">
    {{< img src="/images/support/callout-icon.svg" alt="Callout Icon" width="32" height="32" >}}
  </div>
  <h2>お探しの情報が見つかりませんか？</h2>
  <a href="mailto:support@wandb.com" className="contact-us-button">
    お問い合わせ
  </a>
 {{< /card >}}
