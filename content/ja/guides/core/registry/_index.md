---
title: レジストリ
menu:
  default:
    identifier: registry
    parent: core
weight: 3
url: guides/core/registry
cascade:
- url: guides/core/registry/:filename
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb_registry/zoo_wandb.ipynb" >}}

W&B Registry は、組織内の [W&B Artifact]({{< relref "/guides/core/artifacts/" >}}) バージョンを一元管理するためのキュレートされたリポジトリです。組織で [権限を持つユーザー]({{< relref "./configure_registry.md" >}}) は、[Artifacts をダウンロード・利用]({{< relref "./download_use_artifact.md" >}}) したり、共有やライフサイクル管理をチームを問わず共同で行うことができます。

Registry を使えば、[Artifact バージョンの追跡]({{< relref "./link_version.md" >}})、Artifact 利用履歴や変更履歴の監査、Artifact のガバナンスやコンプライアンス担保、[モデル CI/CD などの下流プロセスの自動化]({{< relref "/guides/core/automations/" >}}) などが可能です。

まとめると、W&B Registry の主な用途は次の通りです。

- 機械学習タスクを満足した Artifact バージョンを、組織内の他のユーザーへ [Promote（昇格）]({{< relref "./link_version.md" >}}) する。
- [タグを使って Artifacts を整理]({{< relref "./organize-with-tags.md" >}}) し、目的の Artifact を検索・参照しやすくする。
- [Artifact のリネージ（履歴）]({{< relref "/guides/core/registry/lineage.md" >}}) を追跡し、変更履歴を監査する。
- [モデル CI/CD などの下流プロセスを自動化]({{< relref "/guides/core/automations/" >}}) する。
- [組織内で Artifact へのアクセス権限を制御]({{< relref "./configure_registry.md" >}}) する。

{{< img src="/images/registry/registry_landing_page.png" alt="W&B Registry" >}}

上記の画像は、Registry App で "Model" や "Dataset" などコア Registry やカスタム Registry を表示している様子です。

## 基本を学ぶ
各組織には最初から **Models** および **Datasets** という 2 つの Registry が用意されており、各 Registry でモデルやデータセットの Artifact を整理できます。[組織のニーズに応じて新たに Registry を作成し、他のタイプの Artifact も管理可能です]({{< relref "./registry_types.md" >}})。

各 [Registry]({{< relref "./configure_registry.md" >}}) は 1 つ以上の [コレクション]({{< relref "./create_collection.md" >}}) から構成され、各コレクションは特定のタスクやユースケースを表します。

{{< img src="/images/registry/homepage_registry.png" alt="W&B Registry" >}}

Artifact を Registry に追加するには、まず [特定バージョンの Artifact を W&B にログ]({{< relref "/guides/core/artifacts/create-a-new-artifact-version.md" >}}) します。Artifact をログするたびに W&B が自動的にバージョンを割り振ります。バージョンは 0 始まり（0 インデックス）なので、最初は `v0`、次が `v1` という形で増えていきます。

Artifact を W&B にログしたら、その Artifact バージョンを Registry 内のコレクションへリンクできます。

{{% alert %}}
「リンク」とは、W&B が Artifact を保存している場所と、Registry でアクセスできる場所とをポインターで結びつけることを意味します。リンクしても Artifact 本体が複製されることはありません。
{{% /alert %}}

例として、次のコード例では「my_model.txt」というモデルファイルを "first-collection" というコレクションに、[コア Registry]({{< relref "./registry_types.md" >}}) 内でログ＆リンクする手順を示しています。

1. W&B Run を初期化
2. Artifact を W&B にログ
3. リンク先のコレクション名と Registry 名を指定
4. コレクションへ Artifact をリンク

この Python コードをスクリプトとして保存し、実行してください。W&B Python SDK バージョン 0.18.6 以降が必要です。

```python title="hello_collection.py"
import wandb
import random

# Artifact を追跡するための W&B Run を初期化
run = wandb.init(project="registry_quickstart") 

# ログ用のモデルファイルを作成
with open("my_model.txt", "w") as f:
   f.write("Model: " + str(random.random()))

# Artifact を W&B にログ
logged_artifact = run.log_artifact(
    artifact_or_path="./my_model.txt", 
    name="gemma-finetuned", 
    type="model" # Artifact のタイプを指定
)

# コレクション名と Registry 名を指定
# Artifact を公開したいコレクション・Registry 名
COLLECTION_NAME = "first-collection"
REGISTRY_NAME = "model"

# Artifact を Registry にリンク
run.link_artifact(
    artifact=logged_artifact, 
    target_path=f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"
)
```

もし指定したコレクション名が Registry 内に存在しない場合でも、`link_artifact(target_path = "")` メソッドが自動的にコレクションを作成します。

{{% alert %}}
ターミナルに表示される URL は、W&B が Artifact を保存したプロジェクトのページです。 
{{% /alert %}}

Registry App から、自分や他メンバーが公開した Artifact バージョンを確認できます。W&B のトップページから左サイドバーの **Applications** 配下にある **Registry** をクリックし、「Model」レジストリを選択してください。Registry 内で "first-collection" コレクションと、自分がリンクした Artifact バージョンを確認できます。

Artifact バージョンを Registry 内のコレクションにリンクすると、組織内の他のメンバーも適切な権限があれば Artifact バージョンの閲覧、ダウンロード、管理、下流オートメーションの作成などが可能になります。

{{% alert %}}
Artifact バージョンでメトリクス（`run.log_artifact()` などで）をログした場合、そのバージョンの詳細ページからメトリクス閲覧ができます。また、コレクションページでは複数バージョン間でメトリクス比較も可能です。[Registry でリンク済み Artifact を表示する方法はこちら]({{< relref "link_version.md#view-linked-artifacts-in-a-registry" >}})。
{{% /alert %}}

## W&B Registry を有効化する

デプロイメントタイプに応じて、W&B Registry を有効化する方法は以下の通りです。

| デプロイメントタイプ | 有効化手順 |
| ----- | ----- |
| マルチテナントクラウド | 特別な操作は不要です。W&B App で Registry が使用可能です。|
| 専用クラウド | アカウント担当者へご連絡ください。お使いのデプロイメントに Registry を有効化します。|
| セルフマネージド | 環境変数 `ENABLE_REGISTRY_UI` を `true` に設定してください。[環境変数の設定方法はこちら]({{< relref "/guides/hosting/env-vars.md" >}})。サーバー v0.59.2 以降が必要です。|


## はじめに役立つリソース　

ユースケースに合わせて、W&B Registry 利用開始のために以下のリソースをご活用ください。

* チュートリアル動画をご覧ください：
    * [Getting started with Registry from W&B](https://www.youtube.com/watch?v=p4XkVOsjIeM)
* W&B の [Model CI/CD](https://www.wandb.courses/courses/enterprise-model-management) コースで以下を学べます：
    * W&B Registry を使った artifacts の管理・バージョン管理・リネージ追跡・モデルのライフサイクル管理
    * Webhooks を使ったモデル管理ワークフローの自動化
    * 外部の ML システムやツールとの統合によるモデル評価、モニタリング、デプロイメント



## 旧 Model Registry から W&B Registry への移行

旧 Model Registry は廃止予定となっていますが、正確な日程は未定です。廃止前に、旧 Model Registry の内容は W&B Registry へ移行されます。

旧 Model Registry から新しい W&B Registry への移行について詳しくは、[こちら]({{< relref "./model_registry_eol.md" >}}) をご覧ください。

移行が完了するまでは、旧 Model Registry と新しい Registry の両方が引き続き利用可能です。

{{% alert %}}
旧 Model Registry を見るには、W&B App 内の Model Registry を開いてください。ページ上部にバナーが表示され、旧 Model Registry App UI を使うことができます。

{{< img src="/images/registry/nav_to_old_model_reg.gif" alt="Legacy Model Registry UI" >}}
{{% /alert %}}


ご不明点やご要望があれば、support@wandb.com までお気軽にお問い合わせください。W&B プロダクトチームとも直接ご相談いただけます。