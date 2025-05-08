---
title: レジストリ
cascade:
- url: /ja/guides/core/registry/:filename
menu:
  default:
    identifier: ja-guides-core-registry-_index
    parent: core
url: /ja/guides/core/registry
weight: 3
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb_registry/zoo_wandb.ipynb" >}}

{{% alert %}}
W&B Registryは現在パブリックプレビュー中です。有効化の方法については、[こちら]({{< relref path="./#enable-wb-registry" lang="ja" >}})のセクションを参照してください。
{{% /alert %}}

W&B Registryは、組織内のアーティファクトバージョンの管理された中央リポジトリです。組織内で[権限を持つ]({{< relref path="./configure_registry.md" lang="ja" >}})ユーザーは、すべてのアーティファクトのライフサイクルを[ダウンロード]({{< relref path="./download_use_artifact.md" lang="ja" >}})、共有、共同管理できます。  
 
Registryを使用して、[アーティファクトバージョンを追跡]({{< relref path="./link_version.md" lang="ja" >}})し、アーティファクトの使用履歴と変更履歴を監査し、アーティファクトのガバナンスとコンプライアンスを保証し、[モデルCI/CDなどの下流プロセスを自動化]({{< relref path="/guides/core/automations/" lang="ja" >}})できます。

要約すると、W&B Registryを使用して以下を行うことができます：

- 機械学習タスクを満たすアーティファクトバージョンを組織内の他のユーザーに[プロモート]({{< relref path="./link_version.md" lang="ja" >}})します。
- [アーティファクトをタグで整理]({{< relref path="./organize-with-tags.md" lang="ja" >}})し、特定のアーティファクトを見つけたり参照したりします。
- [アーティファクトのリネージ]({{< relref path="/guides/core/registry/lineage.md" lang="ja" >}})を追跡し、変更履歴を監査します。
- モデルCI/CDなどの[下流プロセスを自動化]({{< relref path="/guides/core/automations/" lang="ja" >}})します。
- 組織内の誰が各レジストリのアーティファクトに[アクセスできるかを制限]({{< relref path="./configure_registry.md" lang="ja" >}})します。

{{< img src="/images/registry/registry_landing_page.png" alt="" >}}

前の画像は「Model」と「Dataset」のコアレジストリとカスタムレジストリがあるRegistryアプリを示しています。

## 基本を学ぶ

各組織には最初にモデルとデータセットのアーティファクトを整理するための**Models**と**Datasets**と呼ばれる2つのレジストリが含まれています。組織のニーズに基づいて、[他のアーティファクトタイプを整理するための追加のレジストリを作成する]({{< relref path="./registry_types.md" lang="ja" >}})ことができます。

[レジストリ]({{< relref path="./configure_registry.md" lang="ja" >}})は1つ以上の[コレクション]({{< relref path="./create_collection.md" lang="ja" >}})で構成されています。各コレクションは、異なるタスクまたはユースケースを表しています。

{{< img src="/images/registry/homepage_registry.png" >}}

レジストリにアーティファクトを追加するには、最初に[特定のアーティファクトバージョンをW&Bにログ]({{< relref path="/guides/core/artifacts/create-a-new-artifact-version.md" lang="ja" >}})します。アーティファクトをログすると、W&Bは自動的にそのアーティファクトにバージョンを割り当てます。アーティファクトバージョンは0から始まり、最初のバージョンは`v0`、2番目は`v1`というように続きます。

アーティファクトをW&Bにログしたら、その特定のアーティファクトバージョンをレジストリ内のコレクションにリンクできます。

{{% alert %}}
「リンク」という用語は、W&Bがアーティファクトを保存している場所と、レジストリでアーティファクトにアクセスできる場所を接続するポインタを指します。アーティファクトをコレクションにリンクするときにW&Bはアーティファクトを重複しません。
{{% /alert %}}

例として、以下のコード例では、"my_model.txt"という名前のモデルアーティファクトを[コアレジストリ]({{< relref path="./registry_types.md" lang="ja" >}})内の"first-collection"というコレクションにログしリンクする方法を示しています：

1. W&Bのrunを初期化します。
2. アーティファクトをW&Bにログします。
3. コレクションとレジストリの名前を指定して、アーティファクトバージョンをリンクします。
4. アーティファクトをコレクションにリンクします。

このPythonコードをスクリプトとして保存し実行してください。W&B Python SDKバージョン0.18.6以上が必要です。

```python title="hello_collection.py"
import wandb
import random

# アーティファクトをトラックするためにW&Bのrunを初期化
run = wandb.init(project="registry_quickstart") 

# シミュレートされたモデルファイルを作成してログします
with open("my_model.txt", "w") as f:
   f.write("Model: " + str(random.random()))

# アーティファクトをW&Bにログします
logged_artifact = run.log_artifact(
    artifact_or_path="./my_model.txt", 
    name="gemma-finetuned", 
    type="model" # アーティファクトタイプを指定
)

# 公開したいコレクションとレジストリの名前を指定
COLLECTION_NAME = "first-collection"
REGISTRY_NAME = "model"

# アーティファクトをレジストリにリンク
run.link_artifact(
    artifact=logged_artifact, 
    target_path=f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"
)
```

指定したレジストリ内に存在しない場合、`link_artifact(target_path = "")`メソッドで返されるrunオブジェクトの指定したコレクションをW&Bが自動的に作成します。

{{% alert %}}
ターミナルが出力するURLは、W&Bがあなたのアーティファクトを保存しているプロジェクトにリダイレクトされます。
{{% /alert %}}

他の組織メンバーと公開したアーティファクトバージョンを表示するためにRegistryアプリに移動します。まずW&Bに移動します。左側のサイドバーで**Applications**以下の**Registry**を選択します。「Model」レジストリを選択します。そのレジストリ内で、リンクしたアーティファクトバージョンを持つ「first-collection」コレクションが表示されるはずです。

アーティファクトバージョンをレジストリ内のコレクションにリンクすると、それを所有している組織メンバーは、適切な権限を持っていれば、あなたのアーティファクトバージョンを表示、ダウンロード、管理し、下流のオートメーションを作成できます。

{{% alert %}}
もしアーティファクトバージョンがメトリクスをログする場合（たとえば`run.log_artifact()`を使用して）、そのバージョンの詳細ページからメトリクスを表示できますし、コレクションのページからアーティファクトバージョン間でメトリクスを比較できます。参照：[レジストリ内のリンクされたアーティファクトを表示する]({{< relref path="link_version.md#view-linked-artifacts-in-a-registry" lang="ja" >}})。
{{% /alert %}}

## W&B Registryを有効にする

デプロイメントタイプに基づいて、以下の条件を満たしてW&B Registryを有効にします：

| デプロイメントタイプ | 有効にする方法 |
| ----- | ----- |
| マルチテナントクラウド | アクションは必要ありません。W&B RegistryはW&Bアプリで利用可能です。 |
| 専用クラウド | アカウントチームに連絡してください。ソリューションアーキテクト(SA)チームがあなたのインスタンスのオペレーターコンソール内でW&B Registryを有効にします。インスタンスがサーバーリリースバージョン0.59.2以上であることを確認してください。|
| セルフマネージド   | `ENABLE_REGISTRY_UI`と呼ばれる環境変数を有効にします。サーバーで環境変数を有効にする方法の詳細については[こちらのドキュメント]({{< relref path="/guides/hosting/env-vars/" lang="ja" >}})をご覧ください。セルフマネージドインスタンスでは、インフラストラクチャ管理者がこの環境変数を有効にして`true`に設定する必要があります。インスタンスがサーバーリリースバージョン0.59.2以上であることを確認してください。|

## 開始するためのリソース

ユースケースに応じて、W&B Registryの利用を開始するための以下のリソースを探索してください：

* チュートリアルビデオをチェックしてください：
    * [Weights & BiasesからのRegistryの開始方法](https://www.youtube.com/watch?v=p4XkVOsjIeM)
* W&Bの[Model CI/CD](https://www.wandb.courses/courses/enterprise-model-management) コースを受講し、以下を学びましょう：
    * W&B Registryを使用してアーティファクトを管理し、バージョン管理、リネージトラッキング、および異なるライフサイクルステージへのモデルプロモートを行います。
    * Webhooksを使用してモデル管理ワークフローを自動化します。
    * 外部MLシステムおよびツールとレジストリを統合して、モデル評価、モニタリング、デプロイメントを行います。

## 旧モデルレジストリからW&B Registryへの移行

旧モデルレジストリの廃止予定日は未定です。旧モデルレジストリを廃止する前に、W&Bは旧モデルレジストリの内容をW&B Registryに移行します。

旧モデルレジストリからW&B Registryへの移行プロセスについて詳しくは、[旧モデルレジストリからの移行]({{< relref path="./model_registry_eol.md" lang="ja" >}})を参照してください。

移行が行われるまで、W&Bは旧モデルレジストリと新しいレジストリの両方をサポートしています。

{{% alert %}}
旧モデルレジストリを表示するには、W&BアプリでModel Registryに移動してください。ページの上部に、旧モデルレジストリアプリUIを使用できるバナーが表示されます。

{{< img src="/images/registry/nav_to_old_model_reg.gif" alt="" >}}
{{% /alert %}}

質問がある場合や移行についての懸念がある場合は、support@wandb.comにお問い合わせいただくか、W&Bプロダクトチームとお話しください。