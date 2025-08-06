---
title: 自動化イベントとスコープ
menu:
  default:
    identifier: automation-scopes
    parent: automations
weight: 2
---

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

オートメーションは、プロジェクトやレジストリ内で特定のイベントが発生したときに開始することができます。このページでは、各スコープでオートメーションをトリガーできるイベントについて説明します。オートメーションの詳細については、[オートメーション概要]({{< relref "/guides/core/automations/" >}})または[オートメーションの作成方法]({{< relref "create-automations/" >}})をご覧ください。

## Registry
このセクションでは、[Registry]({{< relref "/guides/core/registry/">}})のオートメーションで利用できるスコープとイベントについて説明します。

1. https://wandb.ai/registry/ の **Registry** アプリにアクセスします。
1. Registry の名前をクリックし、**Automations** タブでオートメーションを表示・作成できます。

![Registry Automations タブのオートメーションのスクリーンショット](/images/automations/registry_automations_tab.png)

[オートメーションの作成方法]({{< relref "create-automations/" >}})の詳細もご確認ください。

### スコープ
Registry のオートメーションは、以下のスコープで作成できます。
- [Registry]({{< relref "/guides/core/registry/">}}) レベル: オートメーションは、特定の registry 内にあるすべてのコレクション（将来追加されるものも含む）のイベントを監視します。
- コレクションレベル: 特定の registry 内の単一コレクション。

### イベント
Registry のオートメーションは、次のイベントを監視できます。
- **新しいバージョンがコレクションにリンクされる**: Registry に新しいモデルやデータセットが追加された際にテストや検証を自動化できます。
- **artifact エイリアスが追加される**: 特定のエイリアスが適用された新しい artifact バージョンが発生した際、ワークフローの特定ステップをトリガーします。例: `production` エイリアスが適用されたらモデルをデプロイする。

## Project
このセクションでは、[project]({{< relref "/guides/models/track/project-page.md" >}})のオートメーションで利用できるスコープとイベントについて説明します。

1. W&B アプリの `https://wandb.ai/<team>/<project-name>` で自分のプロジェクトにアクセスします。
1. **Automations** タブでオートメーションを表示・作成できます。

![Project Automations タブのオートメーションのスクリーンショット](/images/automations/project_automations_tab.png)

[オートメーションの作成方法]({{< relref "create-automations/" >}})の詳細もご確認ください。

### スコープ
プロジェクトのオートメーションは、以下のスコープで作成できます。
- プロジェクトレベル: プロジェクト内のすべてのコレクションで発生したイベントを監視します。
- コレクションレベル: 指定したフィルターに一致するプロジェクト内のすべてのコレクション。

### Artifact に関するイベント
ここでは、artifact 関連のイベントについて説明します。これらがオートメーションのトリガーとなります。

- **artifact に新しいバージョンが追加される**: artifact の各バージョンに対して反復的なアクションを適用できます。例: 新しいデータセット artifact バージョンが作成されたらトレーニングジョブを開始します。
- **artifact エイリアスが追加される**: プロジェクトやコレクションで特定のエイリアスが適用された新しい artifact バージョンが発生した際、ワークフローの特定ステップをトリガーします。例: `test-set-quality-check` エイリアスが適用されたときに下流のプロセッシングステップを実行したり、`latest` エイリアスが新バージョンについたたびにワークフローを実行します。同時に1つの artifact バージョンだけが、あるエイリアスを持てます。
- **artifact タグが追加される**: プロジェクトやコレクションの artifact バージョンに特定のタグが適用されたときにワークフローの特定ステップをトリガーします。例: タグ「europe」が artifact バージョンに追加されたときに地域別ワークフローを実行します。artifact タグはグルーピングやフィルタリング用で、同じタグを複数の artifact バージョンに同時に割り当てることができます。

### Run に関するイベント
[run のステータス]({{< relref "/guides/models/track/runs/#run-states" >}})の変更や、[メトリクスの値]({{< relref "/guides/models/track/log/#what-data-is-logged-with-specific-wb-api-calls" >}})の変化によってオートメーションをトリガーすることも可能です。

#### run ステータスの変更
{{% alert %}}
- 現在、[W&B Multi-tenant Cloud]({{< relref "/guides/hosting/#wb-multi-tenant-cloud" >}}) のみで利用可能です。
- **Killed** ステータスの run はオートメーションをトリガーできません。このステータスは、run が管理者ユーザーによって強制停止されたことを示します。
{{% /alert %}}

run がその [ステータス]({{< relref "/guides/models/track/runs/_index.md#run-states" >}})を **Running**、**Finished**、または **Failed** に変更したときにワークフローをトリガーします。さらに、run を開始したユーザーや run 名でフィルターをかけてトリガーする run を限定することもできます。

![run ステータス変更オートメーションのスクリーンショット](/images/automations/run_status_change.png)

run ステータスは run 全体のプロパティのため、run ステータスによるオートメーションは **Automations** ページからのみ作成でき、ワークスペースからは作成できません。

#### run メトリクスの変化
{{% alert %}}
現在、[W&B Multi-tenant Cloud]({{< relref "/guides/hosting/#wb-multi-tenant-cloud" >}}) のみで利用可能です。
{{% /alert %}}

run の履歴に記録されたメトリクス、または `cpu`（CPU 使用率のパーセンテージなど）のような [システムメトリクス]({{< relref "/guides/models/app/settings-page/system-metrics.md" >}}) の値によってワークフローをトリガーします。W&B ではシステムメトリクスが15秒ごとに自動記録されます。

run メトリクスオートメーションは、プロジェクトの **Automations** タブ、またはワークスペース内の折れ線グラフパネルから直接作成できます。

run メトリックオートメーションのセットアップ時は、メトリクスの値を指定した閾値とどのように比較するかを設定します。選択肢はイベントタイプや指定したフィルターによって異なります。

さらに、run を開始したユーザーや run 名でフィルターして、オートメーションをトリガーする対象 run を限定することも可能です。

##### 閾値
**Run metrics threshold met** イベントでは、次のように設定します:
1. 対象とする最新の記録値ウィンドウ（デフォルトは5）。
1. そのウィンドウで **Average**、**Min**、**Max** のどれを評価するか。
1. 比較方法
      - より大きい
      - 以上
      - より小さい
      - 以下
      - 等しくない
      - 等しい

例: 平均 `accuracy` が `.6` を超えたときにオートメーションをトリガーする。

![run メトリクス閾値オートメーションのスクリーンショット](/images/automations/run_metrics_threshold_automation.png)

##### 変化の閾値
**Run metrics change threshold met** イベントでは、2つの「ウィンドウ」を使って値の変化を確認します。

- _現在のウィンドウ_: 最近ログされた値（デフォルトは10）
- _直前ウィンドウ_: その直前にログされた値（デフォルトは50）

現在と直前のウィンドウは連続しており、重複はありません。

オートメーション作成時の設定内容は次の通りです:
1. 現在のウィンドウ値（デフォルト10）
1. 直前のウィンドウ値（デフォルト50）
1. 値の比較方法が割合（**Relative**がデフォルト）か絶対値か
1. 比較方法
      - 最低限これだけ増加
      - 最低限これだけ減少
      - 最低限増減

例: 平均 `loss` が `.25` 以上減少したらオートメーションをトリガーする。

![run メトリクス変化閾値オートメーションのスクリーンショット](/images/automations/run_metrics_change_threshold_automation.png)

#### run フィルター
このセクションでは、オートメーションが run をどのように選択して評価するかを説明します。

- デフォルトでは、プロジェクト内の任意の run でイベントが発生した際にオートメーションが発動します。特定の run だけ考慮したい場合は run フィルターを指定します。
- 各 run は個別に評価され、オートメーションをトリガーする可能性があります。
- 各 run の値は個別のウィンドウに入り、それぞれ閾値と比較されます。
- 24時間以内に、1つのオートメーションが1つの run に対して発火するのは最大1回です。

## 次のステップ
- [Slack オートメーションの作成]({{< relref "create-automations/slack.md" >}})
- [Webhook オートメーションの作成]({{< relref "create-automations/webhook.md" >}})