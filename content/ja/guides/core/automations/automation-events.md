---
title: Automation events and scopes
menu:
  default:
    identifier: ja-guides-core-automations-automation-events
    parent: automations
weight: 2
---

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

オートメーション は、特定のイベントが project または registry のスコープ内で発生したときに開始できます。 project の *スコープ* は、[スコープの技術的な定義を挿入] を指します。このページでは、各スコープ内で オートメーション をトリガーできるイベントについて説明します。

オートメーション の詳細については、[オートメーション の概要]({{< relref path="/guides/core/automations/" lang="ja" >}}) または [オートメーション の作成]({{< relref path="create-automations/" lang="ja" >}}) を参照してください。

## Registry
このセクションでは、[Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) 内の オートメーション のスコープとイベントについて説明します。

1. https://wandb.ai/registry/ で **Registry** App に移動します。
2. registry の名前をクリックし、**Automations** タブで オートメーション を表示および作成します。

[オートメーション の作成]({{< relref path="create-automations/" lang="ja" >}}) の詳細について説明します。

### スコープ
次のスコープで Registry オートメーション を作成できます。
- [Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) レベル: オートメーション は、特定の registry 内のコレクション (今後追加されるコレクションを含む) で発生するイベントを監視します。
- コレクション レベル: 特定の registry 内の単一のコレクション。

### イベント
Registry オートメーション は、次のイベントを監視できます。
- **新しい Artifact をコレクションにリンクする**: registry に追加された新しい Models または Datasets をテストおよび検証します。
- **新しい エイリアス を Artifact の バージョン に追加する**: 新しい Artifact バージョン に特定の エイリアス が適用されたときに、 ワークフロー の特定のステップをトリガーします。たとえば、`production` エイリアス が適用されたときに model をデプロイします。

## Project
このセクションでは、[project]({{< relref path="/guides/models/track/project-page.md" lang="ja" >}}) 内の オートメーション のスコープとイベントについて説明します。

1. W&B App ( `https://wandb.ai/<team>/<project-name>` ) で W&B project に移動します。
2. **Automations** タブで オートメーション を表示および作成します。

[オートメーション の作成]({{< relref path="create-automations/" lang="ja" >}}) の詳細について説明します。

### スコープ
次のスコープで project オートメーション を作成できます。
- Project レベル: オートメーション は、 project 内のコレクションで発生するイベントを監視します。
- コレクション レベル: 指定したフィルターに一致する project 内のすべてのコレクション。

### イベント
project オートメーション は、次のイベントを監視できます。
- **Artifact の新しい バージョン がコレクションに作成される**: Artifact の各 バージョン に定期的なアクションを適用します。コレクション の指定はオプションです。たとえば、新しい dataset Artifact バージョン が作成されたときに training ジョブを開始します。
- **Artifact エイリアス が追加される**: project またはコレクション内の新しい Artifact バージョン に特定の エイリアス が適用されたときに、 ワークフロー の特定のステップをトリガーします。たとえば、Artifact に `test-set-quality-check` エイリアス が適用されたときに、一連のダウンストリーム プロセッシング ステップを実行します。

## 次のステップ
- [Slack オートメーション を作成する]({{< relref path="create-automations/slack.md" lang="ja" >}})
- [Webhook オートメーション を作成する]({{< relref path="create-automations/webhook.md" lang="ja" >}})
