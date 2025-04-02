---
title: Registry
cascade:
- url: guides/core/registry/:filename
menu:
  default:
    identifier: ja-guides-core-registry-_index
    parent: core
url: guides/core/registry
weight: 3
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb_registry/zoo_wandb.ipynb" >}}

{{% alert %}}
W&B Registry は現在パブリックプレビュー中です。[こちらの]({{< relref path="./#enable-wb-registry" lang="ja" >}})セクションで、デプロイメントタイプに対して有効にする方法を学んでください。
{{% /alert %}}

W&B Registry は、組織内の [artifact]( {{< relref path="/guides/core/artifacts/" lang="ja" >}}) バージョンの厳選された中央リポジトリです。組織内で[許可]({{< relref path="./configure_registry.md" lang="ja" >}})を持っている ユーザー は、所属する Teams に関係なく、すべての artifact のライフサイクルを[ダウンロード]({{< relref path="./download_use_artifact.md" lang="ja" >}})、共有、および共同で管理できます。

Registry を使用して、[artifact の バージョン を追跡]({{< relref path="./link_version.md" lang="ja" >}})し、artifact の使用状況と変更の履歴を監査し、artifact のガバナンスとコンプライアンスを確保し、[モデル CI/CD などのダウンストリーム プロセス を自動化]({{< relref path="/guides/core/automations/" lang="ja" >}})できます。

要約すると、W&B Registry を使用して以下を行います。

- 機械学習タスクを満たす artifact バージョン を組織内の他の ユーザー に[プロモート]({{< relref path="./link_version.md" lang="ja" >}})します。
- 特定の artifact を見つけたり参照したりできるように、[タグを使用して artifact を整理]({{< relref path="./organize-with-tags.md" lang="ja" >}})します。
- [artifact の リネージ を追跡]({{< relref path="/guides/core/registry/lineage.md" lang="ja" >}})し、変更の履歴を監査します。
- モデル CI/CD などのダウンストリーム プロセス を[自動化]({{< relref path="/guides/core/automations/" lang="ja" >}})します。
- [組織内で誰が]({{< relref path="./configure_registry.md" lang="ja" >}})各 registry 内の artifact に アクセス できるかを制限します。

{{< img src="/images/registry/registry_landing_page.png" alt="" >}}

上の図は、"Model" および "Dataset" コア registry とカスタム registry を含む Registry App を示しています。

## 基本を学ぶ

各組織には、モデル および データセット artifact を整理するために使用できる **Models** および **Datasets** という 2 つの registry が最初から含まれています。[組織のニーズに基づいて他の artifact タイプ を整理するために、追加の registry を作成]({{< relref path="./registry_types.md" lang="ja" >}})できます。

各[registry]({{< relref path="./configure_registry.md" lang="ja" >}})は、1 つ以上の[コレクション]({{< relref path="./create_collection.md" lang="ja" >}})で構成されています。各コレクションは、明確なタスクまたは ユースケース を表します。

{{< img src="/images/registry/homepage_registry.png" >}}

artifact を registry に追加するには、まず[特定の artifact バージョン を W&B に ログ します]({{< relref path="/guides/core/artifacts/create-a-new-artifact-version.md" lang="ja" >}})。artifact を ログ するたびに、W&B はその artifact に バージョン を自動的に割り当てます。artifact バージョン は 0 からインデックスを開始するため、最初の バージョン は `v0`、2 番目の バージョン は `v1` のようになります。

artifact を W&B に ログ したら、その特定の artifact バージョン を registry 内のコレクションにリンクできます。

{{% alert %}}
「リンク」という用語は、W&B が artifact を保存する場所と、registry 内で artifact に アクセス できる場所とを接続するポインタを指します。artifact をコレクションにリンクしても、W&B は artifact を複製しません。
{{% /alert %}}

例として、次の コード 例は、"my_model.txt" という名前のモデル artifact を[コア registry]({{< relref path="./registry_types.md" lang="ja" >}})の "first-collection" という名前のコレクションに ログ してリンクする方法を示しています。

1. W&B run を初期化します。
2. artifact を W&B に ログ します。
3. artifact バージョン をリンクするコレクションと registry の名前を指定します。
4. artifact をコレクションにリンクします。

この Python コード を スクリプト に保存して実行します。W&B Python SDK バージョン 0.18.6 以降が必要です。

```python title="hello_collection.py"
import wandb
import random

# Artifact を追跡するために W&B run を初期化します。
run = wandb.init(project="registry_quickstart") 

# ログ できるようにシミュレートされたモデルファイルを作成します
with open("my_model.txt", "w") as f:
   f.write("Model: " + str(random.random()))

# Artifact を W&B に ログ します
logged_artifact = run.log_artifact(
    artifact_or_path="./my_model.txt", 
    name="gemma-finetuned", 
    type="model" # artifact タイプを指定します
)

# artifact を公開するコレクションと registry の名前を指定します
COLLECTION_NAME = "first-collection"
REGISTRY_NAME = "model"

# artifact を registry にリンクします
run.link_artifact(
    artifact=logged_artifact, 
    target_path=f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"
)
```

返された run オブジェクトの `link_artifact(target_path = "")` メソッドで指定したコレクションが、指定した registry 内に存在しない場合、W&B は自動的にコレクションを作成します。

{{% alert %}}
ターミナル に表示される URL は、W&B が artifact を保存する プロジェクト に移動します。
{{% /alert %}}

Registry App に移動して、 ユーザー と組織の他のメンバーが公開する artifact バージョン を表示します。これを行うには、まず W&B に移動します。[アプリケーション] の下の左側のサイドバーで **Registry** を選択します。"Model" registry を選択します。registry 内に、リンクされた artifact バージョン を持つ "first-collection" コレクションが表示されます。

artifact バージョン を registry 内のコレクションにリンクすると、組織のメンバーは、適切な権限を持っている場合、artifact バージョン の表示、ダウンロード、管理、ダウンストリーム オートメーション の作成などを行うことができます。

{{% alert %}}
artifact バージョン が (たとえば、`run.log_artifact()` を使用して) メトリクス を ログ する場合、詳細ページからその バージョン の メトリクス を表示したり、コレクションのページから artifact バージョン 全体で メトリクス を比較したりできます。[registry でリンクされた artifact を表示]({{< relref path="link_version.md#view-linked-artifacts-in-a-registry" lang="ja" >}})を参照してください。
{{% /alert %}}

## W&B Registry を有効にする

デプロイメントタイプに基づいて、次の条件を満たして W&B Registry を有効にします。

| デプロイメントタイプ | 有効にする方法 |
| ----- | ----- |
| Multi-tenant Cloud | アクションは不要です。W&B Registry は W&B App で利用できます。 |
| 専用クラウド | アカウントチームにお問い合わせください。ソリューションアーキテクト (SA) チームが、インスタンスのオペレーターコンソール内で W&B Registry を有効にします。インスタンスがサーバーリリースバージョン 0.59.2 以降であることを確認してください。|
| Self-Managed   | `ENABLE_REGISTRY_UI` という名前の 環境変数 を有効にします。サーバー で 環境変数 を有効にする方法の詳細については、[これらのドキュメント]({{< relref path="/guides/hosting/env-vars/" lang="ja" >}})を参照してください。セルフマネージドインスタンスでは、インフラストラクチャ 管理者 がこの 環境変数 を有効にして `true` に設定する必要があります。インスタンスがサーバーリリースバージョン 0.59.2 以降であることを確認してください。|

## 開始するためのリソース

ユースケース に応じて、W&B Registry を開始するための次のリソースをご覧ください。

* チュートリアルビデオをご覧ください。
    * [Weights & Biases から Registry を開始する](https://www.youtube.com/watch?v=p4XkVOsjIeM)
* W&B の [Model CI/CD](https://www.wandb.courses/courses/enterprise-model-management) コースを受講して、次の方法を学びましょう。
    * W&B Registry を使用して、artifact の管理と バージョン 管理、リネージ の追跡、さまざまなライフサイクルステージでのモデル のプロモーションを行います。
    * Webhook を使用して、モデル 管理 ワークフロー を自動化します。
    * registry を外部 ML システム および ツール と統合して、モデル の 評価 、監視、および デプロイメント を行います。

## レガシー Model Registry から W&B Registry への移行

レガシー Model Registry は、正確な日付はまだ決定されていませんが、廃止が予定されています。レガシー Model Registry を廃止する前に、W&B はレガシー Model Registry のコンテンツを W&B Registry に移行します。

レガシー Model Registry から W&B Registry への移行 プロセス の詳細については、[レガシー Model Registry からの移行]({{< relref path="./model_registry_eol.md" lang="ja" >}})を参照してください。

移行が行われるまで、W&B はレガシー Model Registry と新しい Registry の両方をサポートします。

{{% alert %}}
レガシー Model Registry を表示するには、W&B App の Model Registry に移動します。ページの上部に、レガシー Model Registry App UI の使用を有効にするバナーが表示されます。

{{< img src="/images/registry/nav_to_old_model_reg.gif" alt="" >}}
{{% /alert %}}

ご質問がある場合、または移行に関する懸念について W&B Product Team と話したい場合は、support@wandb.com までご連絡ください。
