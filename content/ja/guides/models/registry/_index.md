---
title: Registry
cascade:
- url: guides/registry/:filename
menu:
  default:
    identifier: ja-guides-models-registry-_index
    parent: w-b-models
url: guides/registry
weight: 3
---

```markdown
{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb_registry/zoo_wandb.ipynb" >}}


{{% alert %}}
W&B Registry は現在パブリックプレビュー中です。[こちらの]({{< relref path="./#enable-wb-registry" lang="ja" >}})セクションにアクセスして、デプロイメントタイプに対してこれを有効にする方法を学んでください。
{{% /alert %}}


W&B Registry は、組織内の [artifact]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) バージョンの厳選された中央リポジトリです。組織内で [許可されている]({{< relref path="./configure_registry.md" lang="ja" >}}) ユーザーは、所属する Team に関係なく、すべての Artifacts のライフサイクルを [ダウンロード]({{< relref path="./download_use_artifact.md" lang="ja" >}})、共有、および共同で管理できます。

Registry を使用して、[artifact のバージョンを追跡]({{< relref path="./link_version.md" lang="ja" >}}) し、Artifact の使用状況と変更の履歴を監査し、Artifact のガバナンスとコンプライアンスを確保し、[モデル CI/CD などのダウンストリーム プロセスを自動化]({{< relref path="/guides/models/automations/" lang="ja" >}}) できます。

要約すると、W&B Registry は以下のような用途に使用できます。

- 機械学習タスクを満たす Artifact バージョンを組織内の他の User に [プロモート]({{< relref path="./link_version.md" lang="ja" >}}) する。
- 特定の Artifact を見つけたり参照したりできるように、[タグを使用して Artifacts を整理]({{< relref path="./organize-with-tags.md" lang="ja" >}}) する。
- [Artifact のリネージ]({{< relref path="/guides/models/registry/lineage.md" lang="ja" >}}) を追跡し、変更の履歴を監査する。
- モデル CI/CD などのダウンストリーム プロセスを [自動化]({{< relref path="/guides/models/automations/model-registry-automations.md" lang="ja" >}}) する。
- [組織内の誰が]({{< relref path="./configure_registry.md" lang="ja" >}}) 各 Registry 内の Artifacts にアクセスできるかを制限する。




{{< img src="/images/registry/registry_landing_page.png" alt="" >}}

上の図は、"Model" および "Dataset" コア Registry とカスタム Registry を備えた Registry App を示しています。


## 基本を学ぶ
各 Organization には、モデルおよび Dataset Artifacts を整理するために使用できる **Models** および **Datasets** という 2 つの Registry が初期状態で含まれています。[組織のニーズに基づいて他の Artifact タイプを整理するために、追加の Registry を作成]({{< relref path="./registry_types.md" lang="ja" >}}) できます。

各 [Registry]({{< relref path="./configure_registry.md" lang="ja" >}}) は、1 つ以上の [コレクション]({{< relref path="./create_collection.md" lang="ja" >}}) で構成されています。各コレクションは、個別のタスクまたはユースケースを表します。

{{< img src="/images/registry/homepage_registry.png" >}}

Artifact を Registry に追加するには、まず [特定の Artifact バージョンを W&B に記録]({{< relref path="/guides/core/artifacts/create-a-new-artifact-version.md" lang="ja" >}}) します。Artifact を記録するたびに、W&B はその Artifact に自動的にバージョンを割り当てます。Artifact のバージョンは 0 からインデックスが付けられるため、最初のバージョンは `v0`、2 番目のバージョンは `v1` のようになります。

Artifact を W&B に記録したら、その特定の Artifact バージョンを Registry 内のコレクションにリンクできます。

{{% alert %}}
「リンク」という用語は、W&B が Artifact を保存する場所と、Registry 内で Artifact にアクセスできる場所とを接続するポインターを指します。Artifact をコレクションにリンクしても、W&B は Artifact を複製しません。
{{% /alert %}}

例として、次のコード例は、偽のモデル Artifact "my_model.txt" を [コア Model Registry]({{< relref path="./registry_types.md" lang="ja" >}}) の "first-collection" という名前のコレクションに記録およびリンクする方法を示しています。具体的には、このコードは次のことを実現します。

1. Artifact を追跡するために W&B run を初期化します。
2. Artifact を W&B に記録します。
3. Artifact バージョンをリンクするコレクションと Registry の名前を指定します。
4. Artifact をコレクションにリンクします。

次のコードスニペットを Python スクリプトにコピーして実行します。W&B Python SDK のバージョンが 0.18.6 以上であることを確認してください。

```python title="hello_collection.py"
import wandb
import random

# Initialize a W&B run to track the artifact
run = wandb.init(project="registry_quickstart") 

# Create a simulated model file so that you can log it
with open("my_model.txt", "w") as f:
   f.write("Model: " + str(random.random()))

# Log the artifact to W&B
logged_artifact = run.log_artifact(
    artifact_or_path="./my_model.txt", 
    name="gemma-finetuned", 
    type="model" # Specifies artifact type
)

# Specify the name of the collection and registry
# you want to publish the artifact to
COLLECTION_NAME = "first-collection"
REGISTRY_NAME = "model"

# Link the artifact to the registry
run.link_artifact(
    artifact=logged_artifact, 
    target_path=f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"
)
```

W&B は、返された run オブジェクトの `link_artifact(target_path = "")` メソッドで指定したコレクションが、指定した Registry 内に存在しない場合、自動的にコレクションを作成します。

{{% alert %}}
ターミナルが出力する URL は、W&B が Artifact を保存する Project に移動します。
{{% /alert %}}

Registry App に移動して、あなたと組織の他のメンバーが公開した Artifact バージョンを表示します。これを行うには、まず W&B に移動します。[**アプリケーション**] の下の左側のサイドバーで [**Registry**] を選択します。[Model] Registry を選択します。Registry 内に、リンクされた Artifact バージョンを持つ "first-collection" コレクションが表示されます。

Artifact バージョンを Registry 内のコレクションにリンクすると、組織のメンバーは、適切な権限を持っている場合、Artifact バージョンを表示、ダウンロード、および管理したり、ダウンストリームの自動化を作成したりできます。

## W&B Registry を有効にする

デプロイメントタイプに基づいて、次の条件を満たして W&B Registry を有効にします。

| デプロイメントタイプ | 有効にする方法 |
| ----- | ----- |
| Multi-tenant Cloud | アクションは不要です。W&B Registry は W&B App で使用できます。 |
| 専用クラウド | アカウントチームにお問い合わせください。ソリューションアーキテクト（SA）チームは、インスタンスのオペレーターコンソール内で W&B Registry を有効にします。インスタンスがサーバーリリースバージョン 0.59.2 以降であることを確認してください。|
| Self-Managed   | `ENABLE_REGISTRY_UI` という名前の環境変数を有効にします。サーバーで環境変数を有効にする方法の詳細については、[これらのドキュメント]({{< relref path="/guides/hosting/env-vars/" lang="ja" >}})を参照してください。自己管理インスタンスでは、インフラストラクチャ管理者はこの環境変数を有効にして `true` に設定する必要があります。インスタンスがサーバーリリースバージョン 0.59.2 以降であることを確認してください。|


## はじめに役立つリソース

ユースケースに応じて、次のリソースを調べて W&B Registry を使い始めてください。

* チュートリアルビデオをご覧ください。
    * [Weights & Biases からの Registry の使用開始](https://www.youtube.com/watch?v=p4XkVOsjIeM)
* W&B の [Model CI/CD](https://www.wandb.courses/courses/enterprise-model-management) コースを受講して、次の方法を学びます。
    * W&B Registry を使用して Artifacts の管理とバージョン管理、リネージの追跡、ライフサイクルのさまざまな段階でのモデルのプロモーションを行います。
    * Webhook を使用して、モデル管理のワークフローを自動化します。
    * Registry を外部の ML システムおよびツールと統合して、モデルの評価、監視、およびデプロイメントを行います。



## レガシー Model Registry から W&B Registry への移行

レガシー Model Registry は、正確な日付はまだ決定されていませんが、廃止される予定です。レガシー Model Registry を廃止する前に、W&B はレガシー Model Registry のコンテンツを W&B Registry に移行します。


レガシー Model Registry から W&B Registry への移行プロセスの詳細については、[レガシー Model Registry からの移行]({{< relref path="./model_registry_eol.md" lang="ja" >}})を参照してください。

移行が完了するまで、W&B はレガシー Model Registry と新しい Registry の両方をサポートします。

{{% alert %}}
レガシー Model Registry を表示するには、W&B App の Model Registry に移動します。ページの上部に、レガシー Model Registry App UI を使用できるようにするバナーが表示されます。

{{< img src="/images/registry/nav_to_old_model_reg.gif" alt="" >}}
{{% /alert %}}


ご質問がある場合、または移行に関する懸念について W&B Product Team と話したい場合は、support@wandb.com までご連絡ください。
```