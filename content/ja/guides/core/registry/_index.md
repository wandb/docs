---
title: レジストリ
aliases:
- /guides/core/registry/
cascade:
- url: guides/registry/:filename
menu:
  default:
    identifier: ja-guides-core-registry-_index
    parent: core
url: guides/registry
weight: 3
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb_registry/zoo_wandb.ipynb" >}}

W&B Registry は、組織内の [W&B Artifact]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) バージョンをキュレートした中央リポジトリです。組織内で [権限を持つ]({{< relref path="./configure_registry.md" lang="ja" >}}) ユーザーは、所属チームに関係なく、すべてのアーティファクトの [ダウンロードと使用]({{< relref path="./download_use_artifact.md" lang="ja" >}})、共有、ライフサイクル管理を共同で行うことができます。

Registry を使用して、[アーティファクトのバージョンを追跡]({{< relref path="./link_version.md" lang="ja" >}}) し、アーティファクトの使用履歴と変更履歴を監査し、アーティファクトのガバナンスとコンプライアンスを確保し、[モデル CI/CD などのダウンストリーム プロセスを自動化]({{< relref path="/guides/core/automations/" lang="ja" >}}) できます。

要するに、W&B Registry は次の目的で使用します。

- 機械学習タスクを満たすアーティファクト バージョンを組織内の他のユーザーに [プロモート]({{< relref path="./link_version.md" lang="ja" >}}) します。
- 特定のアーティファクトを検索または参照できるように、[タグを使用してアーティファクトを整理]({{< relref path="./organize-with-tags.md" lang="ja" >}}) します。
- [アーティファクトのリネージを追跡]({{< relref path="/guides/core/registry/lineage.md" lang="ja" >}}) し、変更履歴を監査します。
- モデル CI/CD などのダウンストリーム プロセスを [自動化]({{< relref path="/guides/core/automations/" lang="ja" >}}) します。
- 各レジストリでアーティファクトにアクセスできる [組織内のユーザーを制限]({{< relref path="./configure_registry.md" lang="ja" >}}) します。

{{< img src="/images/registry/registry_landing_page.png" alt="W&B Registry" >}}

上記の画像は、「Model」および「Dataset」コア レジストリとカスタム レジストリを備えた Registry アプリを示しています。

## 基本を学ぶ
各組織には最初に、モデルおよびデータセットのアーティファクトを整理するために使用できる「**Models**」と「**Datasets**」という 2 つのレジストリが含まれています。組織のニーズに基づいて、[他のアーティファクト タイプを整理するための追加のレジストリ]({{< relref path="./registry_types.md" lang="ja" >}}) を作成できます。

各 [レジストリ]({{< relref path="./configure_registry.md" lang="ja" >}}) は、1 つ以上の [コレクション]({{< relref path="./create_collection.md" lang="ja" >}}) で構成されます。各コレクションは、個別のタスクまたはユースケースを表します。

{{< img src="/images/registry/homepage_registry.png" alt="W&B Registry" >}}

レジストリにアーティファクトを追加するには、まず [特定のアーティファクト バージョンを W&B にログ]({{< relref path="/guides/core/artifacts/create-a-new-artifact-version.md" lang="ja" >}}) します。アーティファクトをログするたびに、W&B はそのアーティファクトにバージョンを自動的に割り当てます。アーティファクトのバージョンは 0 から始まるインデックスを使用するため、最初のバージョンは `v0`、2 番目のバージョンは `v1` となります。

アーティファクトを W&B にログすると、その特定のアーティファクト バージョンをレジストリ内のコレクションにリンクできます。

{{% alert %}}
「リンク」という用語は、W&B がアーティファクトを保存する場所と、レジストリでアーティファクトにアクセスできる場所を接続するポインタを指します。アーティファクトをコレクションにリンクしても、W&B はアーティファクトを複製しません。
{{% /alert %}}

例として、以下のコード例は、「my_model.txt」というモデル アーティファクトを [コア レジストリ]({{< relref path="./registry_types.md" lang="ja" >}}) の「first-collection」という名前のコレクションにログしてリンクする方法を示しています。

1. W&B Run を初期化します。
2. アーティファクトを W&B にログします。
3. アーティファクト バージョンをリンクするコレクションとレジストリの名前を指定します。
4. アーティファクトをコレクションにリンクします。

この Python コードをスクリプトとして保存し、実行します。W&B Python SDK バージョン 0.18.6 以降が必要です。

```python title="hello_collection.py"
import wandb
import random

# アーティファクトを追跡するために W&B Run を初期化
run = wandb.init(project="registry_quickstart") 

# ログできるようにモデルファイルをダミーで作成
with open("my_model.txt", "w") as f:
   f.write("Model: " + str(random.random()))

# アーティファクトを W&B にログする
logged_artifact = run.log_artifact(
    artifact_or_path="./my_model.txt", 
    name="gemma-finetuned", 
    type="model" # アーティファクトの種類を指定
)

# コレクション名とレジストリ名を指定
# アーティファクトを公開したい対象
COLLECTION_NAME = "first-collection"
REGISTRY_NAME = "model"

# アーティファクトをレジストリにリンクする
run.link_artifact(
    artifact=logged_artifact, 
    target_path=f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"
)
```

指定したレジストリ内に、返された run オブジェクトの `link_artifact(target_path = "")` メソッドで指定したコレクションが存在しない場合、W&B は自動的にそのコレクションを作成します。

{{% alert %}}
ターミナルに表示される URL は、W&B がアーティファクトを保存するプロジェクトに移動します。
{{% /alert %}}

Registry アプリに移動して、あなたや組織の他のメンバーが公開したアーティファクト バージョンを表示します。これを行うには、まず W&B に移動します。「**Applications**」の下にある左側のサイドバーで「**Registry**」を選択します。「Model」レジストリを選択します。レジストリ内に、リンクされたアーティファクト バージョンを持つ「first-collection」コレクションが表示されるはずです。

アーティファクト バージョンをレジストリ内のコレクションにリンクすると、組織のメンバーは適切な権限を持っている場合、そのアーティファクト バージョンを表示、ダウンロード、管理し、ダウンストリームのオートメーションを作成することができます。

{{% alert %}}
アーティファクト バージョンがメトリクスをログしている場合 (例: `run.log_artifact()` を使用している場合)、そのバージョンの詳細ページからメトリクスを表示でき、コレクションのページからアーティファクト バージョン間でメトリクスを比較できます。[レジストリ内のリンクされたアーティファクトを表示する]({{< relref path="link_version.md#view-linked-artifacts-in-a-registry" lang="ja" >}}) を参照してください。
{{% /alert %}}

## W&B Registry を有効にする

デプロイメント タイプに基づいて、W&B Registry を有効にするには、次の条件を満たしてください。

| デプロイメント タイプ | 有効化の方法 |
| ----- | ----- |
| Multi-tenant Cloud | アクションは不要です。W&B Registry は W&B アプリで利用できます。 |
| Dedicated Cloud | デプロイメントで W&B Registry を有効にするには、アカウント チームにお問い合わせください。 |
| Self-Managed | 環境変数 `ENABLE_REGISTRY_UI` を `true` に設定します。[環境変数の設定]({{< relref path="/guides/hosting/env-vars.md" lang="ja" >}}) を参照してください。Server v0.59.2 以降が必要です。 |

## 開始するためのリソース

ユースケースに応じて、W&B Registry の開始に役立つ次のリソースを確認してください。

* チュートリアル ビデオをご覧ください。
    * [Getting started with Registry from W&B](https://www.youtube.com/watch?v=p4XkVOsjIeM)
* W&B の [Model CI/CD](https://www.wandb.courses/courses/enterprise-model-management) コースを受講して、次の方法を学びましょう。
    * W&B Registry を使用して、アーティファクトの管理とバージョン管理、リネージの追跡、異なるライフサイクル段階でのモデルのプロモートを行います。
    * Webhook を使用して、モデル管理ワークフローを自動化します。
    * レジストリを外部の ML システムやツールと統合し、モデルの評価、監視、デプロイメントを行います。

## 旧モデルレジストリから W&B Registry への移行

旧モデルレジストリは廃止予定であり、正確な日付はまだ決定されていません。旧モデルレジストリを廃止する前に、W&B は旧モデルレジストリのコンテンツを W&B Registry に移行します。

旧モデルレジストリから W&B Registry への移行プロセスについては、[旧モデルレジストリからの移行]({{< relref path="./model_registry_eol.md" lang="ja" >}}) を参照してください。

移行が完了するまで、W&B は旧モデルレジストリと新しい Registry の両方をサポートします。

{{% alert %}}
旧モデルレジストリを表示するには、W&B アプリで Model Registry に移動します。ページの上部に、旧モデルレジストリ アプリ UI を使用できるバナーが表示されます。

{{< img src="/images/registry/nav_to_old_model_reg.gif" alt="Legacy Model Registry UI" >}}
{{% /alert %}}

移行に関するご質問や懸念事項については、support@wandb.com までお問い合わせいただくか、W&B 製品チームにご相談ください。