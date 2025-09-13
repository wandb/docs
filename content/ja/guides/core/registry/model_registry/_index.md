---
title: Registered Models
description: Model Registry は、トレーニングからプロダクションまでの Models のライフサイクルを管理します
aliases:
- /guides/core/registry/model_registry
cascade:
- url: /guides/registry/model_registry/:filename
menu:
  default:
    identifier: ja-guides-core-registry-model_registry-_index
    parent: registry
url: /guides/registry/model_registry
weight: 9
---

{{% alert %}}
W&B は最終的に W&B Model Registry のサポートを停止します。ユーザーは、モデルの Artifacts のバージョンをリンクして共有するために、代わりに [W&B Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) を使用することをお勧めします。W&B Registry は、旧モデルレジストリの機能を拡張します。W&B Registry の詳細については、[Registry docs]({{< relref path="/guides/core/registry/" lang="ja" >}}) を参照してください。
W&B は、旧モデルレジストリにリンクされている既存のモデル Artifacts を、近いうちに新しい W&B Registry に移行します。移行プロセスに関する情報については、[旧モデルレジストリからの移行]({{< relref path="/guides/core/registry/model_registry_eol.md" lang="ja" >}}) を参照してください。
{{% /alert %}}

W&B Model Registry は、機械学習の実践者がプロダクション用候補を公開し、下流チームや関係者によって利用される、チームのトレーニング済みモデルを格納します。これは、ステージングされた/候補モデルを格納し、ステージングに関連するワークフローを管理するために使用されます。

{{< img src="/images/models/model_reg_landing_page.png" alt="Model Registry" >}}

W&B Model Registry を使用すると、以下が可能になります。
* [各機械学習タスクで最適なモデルのバージョンをブックマークします。]({{< relref path="./link-model-version.md" lang="ja" >}})
* 下流のプロセスとモデルの CI/CD を[自動化]({{< relref path="/guides/core/automations/" lang="ja" >}})します。
* モデルのバージョンを、ステージングからプロダクションまで、機械学習のライフサイクルを通じて移行させます。
* モデルのリネージを追跡し、プロダクションモデルへの変更履歴を監査します。

{{< img src="/images/models/models_landing_page.png" alt="Models overview" >}}

## 仕組み
いくつかの簡単な手順で、ステージングされたモデルを追跡および管理します。
1. **モデルのバージョンをログに記録する**: トレーニングスクリプトに数行のコードを追加して、モデルファイルを Artifacts として W&B に保存します。
2. **パフォーマンスを比較する**: ライブチャートで、モデルトレーニングと検証からのメトリクスとサンプル予測を比較します。どのモデルのバージョンが最も優れたパフォーマンスを示したかを確認します。
3. **Registry にリンクする**: Python でプログラム的に、または W&B UI でインタラクティブに、最適なモデルのバージョンを Registered Model にリンクしてブックマークします。

以下のコードスニペットは、モデルを Model Registry にログしてリンクする方法を示しています。
```python
import wandb
import random

# 新しい W&B run を開始します
run = wandb.init(project="models_quickstart")

# モデルのメトリクスをログに記録することをシミュレートします
run.log({"acc": random.random()})

# シミュレートされたモデルファイルを作成します
with open("my_model.h5", "w") as f:
    f.write("Model: " + str(random.random()))

# モデルをログに記録し、Model Registry にリンクします
run.link_model(path="./my_model.h5", registered_model_name="MNIST")

run.finish()
```
4. **モデルの遷移を CI/CD ワークフローに接続する**: 候補モデルをワークフローのステージを通じて遷移させ、Webhook を使用して[下流のアクションを自動化]({{< relref path="/guides/core/automations/" lang="ja" >}})します。

## 開始方法
ユースケースに応じて、W&B Models の使用を開始するための以下のリソースを探索してください。
* 2部構成のビデオシリーズをチェックしてください:
  1. [モデルのログ記録と登録](https://www.youtube.com/watch?si=MV7nc6v-pYwDyS-3&v=ZYipBwBeSKE&feature=youtu.be)
  2. Model Registry における[モデルの利用と下流プロセスの自動化](https://www.youtube.com/watch?v=8PFCrDSeHzw)。
* Datasets の Artifacts を作成、追跡、使用するために利用できる W&B Python SDK コマンドのステップバイステップの概要については、[モデルのウォークスルー]({{< relref path="./walkthrough.md" lang="ja" >}}) をお読みください。
* 以下について学びます:
   * [保護されたモデルとアクセス制御]({{< relref path="./access_controls.md" lang="ja" >}})。
   * [Registry を CI/CD プロセスに接続する方法]({{< relref path="/guides/core/automations/" lang="ja" >}})。
   * 新しいモデルのバージョンが Registered Model にリンクされたときに [Slack 通知]({{< relref path="./notifications.md" lang="ja" >}}) を設定します。
* Model Registry を機械学習のワークフローに統合する方法を学ぶために、[ML Model Registry とは何か？](https://wandb.ai/wandb_fc/model-registry-reports/reports/What-is-an-ML-Model-Registry---Vmlldzo1MTE5MjYx) を確認してください。
* W&B の [Enterprise Model Management](https://www.wandb.courses/courses/enterprise-model-management) コースを受講し、以下について学びましょう:
  * W&B Model Registry を使用してモデルを管理およびバージョン管理し、リネージを追跡し、異なるライフサイクルステージを通じてモデルを昇格させる方法
  * Webhook を使用してモデル管理ワークフローを自動化する方法。
  * モデルの評価、監視、デプロイメントのために、Model Registry がモデル開発ライフサイクルにおける外部の機械学習システムおよびツールとどのように統合されるかを確認します。