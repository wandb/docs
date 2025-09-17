---
title: 自動化イベントおよびスコープ
menu:
  default:
    identifier: ja-guides-core-automations-automation-events
    parent: automations
weight: 2
---

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

オートメーション は、Project または Registry 内で特定のイベントが発生したときに開始できます。このページでは、各スコープ内でオートメーション をトリガーできるイベントについて説明します。[オートメーション の概要]({{< relref path="/guides/core/automations/" lang="ja" >}}) または [オートメーション を作成する]({{< relref path="create-automations/" lang="ja" >}}) で、オートメーション について詳しく学びましょう。

## Registry
このセクションでは、[Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) におけるオートメーション のスコープとイベントについて説明します。

1. https://wandb.ai/registry/ にある **Registry** App に移動します。
1. Registry の名前をクリックし、**Automations** タブでオートメーション を表示および作成します。

![オートメーション が表示されている Registry の Automations タブのスクリーンショット](/images/automations/registry_automations_tab.png)

[オートメーション の作成]({{< relref path="create-automations/" lang="ja" >}}) について詳しく学びましょう。

### スコープ
以下のスコープで Registry オートメーション を作成できます。
- [Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) レベル: オートメーション は、特定の Registry 内の任意のコレクション (今後追加されるコレクションを含む) で発生するイベントを監視します。
- コレクション レベル: 特定の Registry 内の単一のコレクション。

### イベント
Registry オートメーション は、以下のイベントを監視できます。
- **新しい バージョン がコレクションにリンクされる**: 新しい Model や Dataset が Registry に追加されたときに、それらをテストおよび検証します。
- **Artifact エイリアス が追加される**: 新しい Artifact バージョン に特定の エイリアス が適用されたときに、ワークフロー の特定のステップをトリガーします。たとえば、Model に `production` エイリアス が適用されたときにデプロイします。

## Project
このセクションでは、[Project]({{< relref path="/guides/models/track/project-page.md" lang="ja" >}}) におけるオートメーション のスコープとイベントについて説明します。

1. W&B App の `https://wandb.ai/<team>/<project-name>` で W&B Project に移動します。
1. **Automations** タブでオートメーション を表示および作成します。

![オートメーション が表示されている Project の Automations タブのスクリーンショット](/images/automations/project_automations_tab.png)

[オートメーション の作成]({{< relref path="create-automations/" lang="ja" >}}) について詳しく学びましょう。

### スコープ
以下のスコープで Project オートメーション を作成できます。
- Project レベル: オートメーション は、Project 内の任意のコレクションで発生するイベントを監視します。
- コレクション レベル: 指定したフィルターに一致する、Project 内のすべてのコレクション。

### Artifact イベント
このセクションでは、Artifact に関連するオートメーション をトリガーできるイベントについて説明します。

- **新しい バージョン が Artifact に追加される**: Artifact の各 バージョン に定期的なアクションを適用します。たとえば、新しい Dataset Artifact バージョン が作成されたときに、トレーニング ジョブを開始します。
- **Artifact エイリアス が追加される**: Project またはコレクション内の新しい Artifact バージョン に特定の エイリアス が適用されたときに、ワークフロー の特定のステップをトリガーします。たとえば、Artifact に `test-set-quality-check` エイリアス が適用されたときに一連のダウンストリーム プロセッシング ステップを実行したり、新しい Artifact バージョン が `latest` エイリアス を取得するたびに ワークフロー を実行したりします。特定の時点では、1 つの Artifact バージョン のみが特定の エイリアス を持つことができます。
- **Artifact タグが追加される**: Project またはコレクション内の Artifact バージョン に特定のタグが適用されたときに、ワークフロー の特定のステップをトリガーします。たとえば、Artifact バージョン に "europe" タグが追加されたときに、地域固有の ワークフロー をトリガーします。Artifact タグはグループ化とフィルタリングに使用され、特定のタグを複数の Artifact バージョン に同時に割り当てることができます。

### Run イベント
オートメーション は、[run のステータス]({{< relref path="/guides/models/track/runs/#run-states" lang="ja" >}}) の変更、または [メトリクス の値]({{< relref path="/guides/models/track/log/#what-data-is-logged-with-specific-wb-api-calls" lang="ja" >}}) の変更によってトリガーできます。

#### Run ステータスの変更
{{% alert %}}
- 現在、[W&B Multi-tenant Cloud]({{< relref path="/guides/hosting/#wb-multi-tenant-cloud" lang="ja" >}}) でのみ利用可能です。
- **Killed** ステータスの run は、オートメーション をトリガーできません。このステータスは、run が管理者 ユーザー によって強制的に停止されたことを示します。
{{% /alert %}}

run が [ステータス]({{< relref path="/guides/models/track/runs/_index.md#run-states" lang="ja" >}}) を **Running**、**Finished**、**Failed** に変更したときに、ワークフロー をトリガーします。オプションで、run を開始した ユーザー または run の名前でフィルターすることにより、オートメーション をトリガーできる run をさらに制限できます。

![run ステータス変更 オートメーション を示すスクリーンショット](/images/automations/run_status_change.png)

run ステータスは run 全体のプロパティであるため、run ステータス オートメーション は **Automations** ページからのみ作成でき、Workspace からは作成できません。

#### Run メトリクス の変更
{{% alert %}}
現在、[W&B Multi-tenant Cloud]({{< relref path="/guides/hosting/#wb-multi-tenant-cloud" lang="ja" >}}) でのみ利用可能です。
{{% /alert %}}

run の履歴内の メトリクス 、または CPU 使用率を追跡する `cpu` などの [システム メトリクス]({{< relref path="/ref/system-metrics.md" lang="ja" >}}) の ログ に基づいて ワークフロー をトリガーします。W&B は、システム メトリクス を 15 秒ごとに自動的に ログ に記録します。

Project の **Automations** タブから、または Workspace の折れ線グラフ パネルから直接、run メトリクス オートメーション を作成できます。

run メトリクス オートメーション を設定するには、メトリクス の 値 を指定したしきい値と比較する方法を設定します。選択肢は、イベントの種類と指定したフィルターによって異なります。

オプションで、run を開始した ユーザー または run の名前でフィルターすることにより、オートメーション をトリガーできる run をさらに制限できます。

##### しきい値
**Run メトリクス のしきい値到達** イベントの場合、以下を設定します。
1. 考慮する最新の ログ に記録された 値 のウィンドウ (デフォルトは 5)。
1. ウィンドウ内の **平均**、**最小**、または **最大** 値 を評価するかどうか。
1. 行う比較:
      - より大きい
      - 以上
      - より小さい
      - 以下
      - 等しくない
      - 等しい

たとえば、平均 `accuracy` が `.6` より大きい場合にオートメーション をトリガーします。

![run メトリクス のしきい値 オートメーション を示すスクリーンショット](/images/automations/run_metrics_threshold_automation.png)

##### 変更しきい値
**Run メトリクス の変更しきい値到達** イベントの場合、オートメーション は 2 つの「ウィンドウ」の 値 を使用して開始するかどうかを確認します。

- 考慮する最新の ログ に記録された 値 の _現在のウィンドウ_ (デフォルトは 10)。
- 考慮する最近の ログ に記録された 値 の _以前のウィンドウ_ (デフォルトは 50)。

現在のウィンドウと以前のウィンドウは連続しており、重複しません。

オートメーション を作成するには、以下を設定します。
1. ログ に記録された 値 の現在のウィンドウ (デフォルトは 10)。
1. ログ に記録された 値 の以前のウィンドウ (デフォルトは 50)。
1. 値 を相対値として評価するか絶対値として評価するか (デフォルトは **相対**)。
1. 行う比較:
      - 少なくとも増加する
      - 少なくとも減少する
      - 少なくとも増減する

たとえば、平均 `loss` が少なくとも `.25` 減少した場合にオートメーション をトリガーします。

![run メトリクス の変更しきい値 オートメーション を示すスクリーンショット](/images/automations/run_metrics_change_threshold_automation.png)

#### Run フィルター
このセクションでは、オートメーション が評価する run を選択する方法について説明します。

- デフォルトでは、Project 内の任意の run は、イベントが発生したときにオートメーション をトリガーします。特定の run のみ考慮するには、run フィルターを指定します。
- 各 run は個別に考慮され、オートメーション をトリガーする可能性があります。
- 各 run の 値 は別のウィンドウに入れられ、しきい値と比較されます。
- 24 時間の期間内で、特定のオートメーション は run ごとに最大 1 回のみトリガーできます。

## 次のステップ
- [Slack オートメーション の作成]({{< relref path="create-automations/slack.md" lang="ja" >}})
- [Webhook オートメーション の作成]({{< relref path="create-automations/webhook.md" lang="ja" >}})