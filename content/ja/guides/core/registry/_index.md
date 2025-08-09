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

W&B Registry は、組織内の [W&B Artifact]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) バージョンを一元管理できるキュレートされたリポジトリです。組織内で [権限を持つユーザー]({{< relref path="./configure_registry.md" lang="ja" >}}) は、[Artifacts のダウンロードと利用]({{< relref path="./download_use_artifact.md" lang="ja" >}})・共有・さまざまなチームを超えた共同管理ができます。

Registry を利用することで、[Artifact のバージョン管理]({{< relref path="./link_version.md" lang="ja" >}})、利用履歴や変更履歴の監査、ガバナンスやコンプライアンス対応、[モデル CI/CD などの下流プロセスの自動化]({{< relref path="/guides/core/automations/" lang="ja" >}}) などが行えます。

まとめると、W&B Registry を使ってできることは次のとおりです：

- 機械学習タスクを満たした Artifact バージョンを他のユーザーに[プロモートする]({{< relref path="./link_version.md" lang="ja" >}})
- [タグで Artifact を整理]({{< relref path="./organize-with-tags.md" lang="ja" >}})し、必要なものを簡単に見つけたり参照したりできる
- [Artifact のリネージ]({{< relref path="/guides/core/registry/lineage.md" lang="ja" >}})の追跡や変更履歴の監査
- [モデル CI/CD などの下流プロセスを自動化する]({{< relref path="/guides/core/automations/" lang="ja" >}})
- 各 Registry への[アクセス制限を管理]({{< relref path="./configure_registry.md" lang="ja" >}})できる

{{< img src="/images/registry/registry_landing_page.png" alt="W&B Registry" >}}

上記の画像は、Registry App の画面例です。"Model" や "Dataset" などのコア Registry のほか、カスタム Registry も表示されています。

## 基本から学ぶ

各組織には最初から **Models** と **Datasets** という 2 つの Registry があり、モデルやデータセットの Artifact を整理できます。 [必要に応じて他の Artifact タイプ向けの Registry を追加作成することも可能です]({{< relref path="./registry_types.md" lang="ja" >}})。

各 [Registry]({{< relref path="./configure_registry.md" lang="ja" >}}) は 1 つ以上の [Collection]({{< relref path="./create_collection.md" lang="ja" >}}) で構成され、それぞれがユースケースやタスクを示します。

{{< img src="/images/registry/homepage_registry.png" alt="W&B Registry" >}}

Registry に Artifact を追加するには、まず [特定の Artifact バージョンを W&B にログ]({{< relref path="/guides/core/artifacts/create-a-new-artifact-version.md" lang="ja" >}}) します。Artifact をログするたびに、W&B が自動でバージョンを割り振ります（0 起算で、最初は `v0`、次が `v1` となります）。

Artifact をログした後、その特定の Artifact バージョンを Registry の Collection にリンクできます。

{{% alert %}}
「リンク」とは、W&B が Artifact を保存している場所と、その Artifact が Registry でアクセスできる場所をつなぐポインタのことを指します。Artifact を Collection にリンクしても、Artifact の複製は発生しません。
{{% /alert %}}

例として、次のコードは "my_model.txt" というモデル Artifact を [コア Registry]({{< relref path="./registry_types.md" lang="ja" >}}) の "first-collection" という Collection にログ＆リンクする方法を示しています。

1. W&B Run を初期化する
2. Artifact を W&B にログする
3. Artifact バージョンをリンクしたい Collection と Registry の名前を指定する
4. Artifact をその Collection にリンクする

この Python コードをスクリプトに保存し、実行してください。W&B Python SDK バージョン 0.18.6 以降が必要です。

```python title="hello_collection.py"
import wandb
import random

# Artifact を追跡するために W&B Run を初期化
run = wandb.init(project="registry_quickstart") 

# ログに使うダミーモデルファイルを作成
with open("my_model.txt", "w") as f:
   f.write("Model: " + str(random.random()))

# Artifact を W&B にログ
logged_artifact = run.log_artifact(
    artifact_or_path="./my_model.txt", 
    name="gemma-finetuned", 
    type="model" # Artifact タイプを指定
)

# Collection と Registry の名前を指定
# この Artifact を公開したい先を設定
COLLECTION_NAME = "first-collection"
REGISTRY_NAME = "model"

# Artifact を Registry にリンク
run.link_artifact(
    artifact=logged_artifact, 
    target_path=f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"
)
```

指定した Collection が Registry 内に存在しなければ、W&B は自動的に Collection を作成します（run オブジェクトの `link_artifact(target_path = "")` メソッドで指定した場合）。

{{% alert %}}
ターミナルに表示される URL は、W&B が Artifact を保存したプロジェクトページへのリンクです。
{{% /alert %}}

Registry App を開くことで、自分や組織のメンバーが公開した Artifact バージョンを閲覧できます。W&B 画面左サイドバーの **Applications** 配下から **Registry** を選択し、"Model" Registry を開いてください。その中に "first-collection" Collection と自分がリンクした Artifact バージョンが表示されているはずです。

Artifact バージョンを Registry 内の Collection にリンクすると、組織の他メンバーは適切な権限があれば Artifact バージョンの閲覧・ダウンロード・管理・オートメーションの作成などが可能になります。

{{% alert %}}
Artifact バージョンがメトリクスをログしている場合（たとえば `run.log_artifact()` など）、そのバージョンの詳細ページからメトリクスを確認できますし、Collection のページで複数 Artifact バージョンのメトリクス比較も行えます。[Registry 内でリンクされた Artifact の閲覧方法についてはこちら]({{< relref path="link_version.md#view-linked-artifacts-in-a-registry" lang="ja" >}}) をご覧ください。
{{% /alert %}}

## W&B Registry を有効化する方法

ご利用のデプロイ方式に応じて、下記の条件を満たせば W&B Registry を有効化できます。

| デプロイ方式 | 有効化方法 |
| ----- | ----- |
| マルチテナントクラウド | 特別な作業不要。W&B Registry は W&B App ですぐに利用できます。|
| 専用クラウド | W&B アカウント担当者へ連絡してください。環境向けに Registry を有効化します。|
| セルフマネージド | 環境変数 `ENABLE_REGISTRY_UI` を `true` に設定してください。[環境変数設定の詳細はこちら]({{< relref path="/guides/hosting/env-vars.md" lang="ja" >}})。Server v0.59.2 以降が必要です。|

## 利用開始に役立つリソース

ユースケースに合わせて、下記リソースで W&B Registry の活用を始めてみましょう。

* チュートリアル動画を見る：
    * [Getting started with Registry from W&B](https://www.youtube.com/watch?v=p4XkVOsjIeM)
* W&B の [Model CI/CD](https://www.wandb.courses/courses/enterprise-model-management) コースを受講し、下記を習得：
    * W&B Registry を使った Artifact の管理・バージョン管理・リネージ追跡・モデルの各ライフサイクルステージのプロモート
    * Webhook を用いたモデル管理ワークフローの自動化
    * 外部 ML システムやツールとの Registry 連携によるモデル評価・モニタリング・デプロイメント

## 旧モデルレジストリから W&B Registry への移行

旧モデルレジストリ（legacy Model Registry）は今後廃止予定ですが、正確な日程は未定です。W&B では、廃止前に旧モデルレジストリ内の内容を新しい W&B Registry へ移行予定です。

移行プロセスの詳細は[旧モデルレジストリからの移行について]({{< relref path="./model_registry_eol.md" lang="ja" >}}) をご参照ください。

移行完了まで、W&B は旧モデルレジストリと新Registryの両方をサポートします。

{{% alert %}}
旧モデルレジストリを見るには、W&B App 内の Model Registry に移動してください。ページ上部に旧モデルレジストリ UI を開くバナーが表示されます。

{{< img src="/images/registry/nav_to_old_model_reg.gif" alt="Legacy Model Registry UI" >}}
{{% /alert %}}

ご質問や移行に関するご相談があれば support@wandb.com までお気軽にご連絡ください。