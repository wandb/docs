---
title: Model registry
description: モデルレジストリを使用して、トレーニングからプロダクションまでのモデルのライフサイクルを管理します。
cascade:
- url: guides/core/registry/model_registry/:filename
menu:
  default:
    identifier: ja-guides-core-registry-model_registry-_index
    parent: registry
url: guides/core/registry/model_registry
weight: 9
---

{{% alert %}}
W&B は最終的に W&B Model Registry のサポートを停止します。 ユーザー は、代わりにモデルアーティファクトの バージョン をリンクおよび共有するために、[W&B Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) を使用することをお勧めします。 W&B Registry は、従来の W&B Model Registry の機能を拡張します。 W&B Registry の詳細については、[Registry のドキュメント]({{< relref path="/guides/core/registry/" lang="ja" >}}) を参照してください。

W&B は、従来の Model Registry にリンクされている既存のモデルアーティファクトを、近い将来、新しい W&B Registry に移行します。 移行 プロセス の詳細については、[従来の Model Registry からの移行]({{< relref path="/guides/core/registry/model_registry_eol.md" lang="ja" >}}) を参照してください。
{{% /alert %}}

W&B Model Registry には、 ML 実践者がプロダクション 用の候補を公開し、ダウンストリーム の チーム や関係者が利用できる、 チーム がトレーニング したモデルが格納されています。 これは、ステージング されたモデル/候補モデルを格納し、ステージング に関連する ワークフロー を管理するために使用されます。

{{< img src="/images/models/model_reg_landing_page.png" alt="" >}}

W&B Model Registry では、次のことが可能です。

* [各 機械学習 タスク に最適なモデル バージョン をブックマークします。]({{< relref path="./link-model-version.md" lang="ja" >}})
* ダウンストリーム プロセス とモデル CI/CD を [オートメーション]({{< relref path="/guides/core/automations/" lang="ja" >}}) します。
* モデル バージョン を ML ライフサイクル (ステージング から プロダクション へ) で移行します。
* モデルの リネージ を追跡し、プロダクション モデルへの変更の履歴を監査します。

{{< img src="/images/models/models_landing_page.png" alt="" >}}

## 仕組み
いくつかの簡単なステップで、ステージング されたモデルを追跡および管理します。

1. **モデル バージョン を ログ に記録する**: トレーニング スクリプト で、数行の コード を追加して、モデルファイルを Artifacts として W&B に保存します。
2. **パフォーマンス を比較する**: ライブ チャート をチェックして、モデル トレーニング と検証からの メトリクス とサンプル 予測 を比較します。 どのモデル バージョン が最も優れた パフォーマンス を発揮したかを特定します。
3. **Registry にリンクする**: Python でプログラムで、または W&B UI でインタラクティブに、登録されたモデルにリンクして、最適なモデル バージョン をブックマークします。

次の コード スニペット は、モデルを ログ に記録して Model Registry にリンクする方法を示しています。

```python
import wandb
import random

# Start a new W&B run
run = wandb.init(project="models_quickstart")

# Simulate logging model metrics
run.log({"acc": random.random()})

# Create a simulated model file
with open("my_model.h5", "w") as f:
    f.write("Model: " + str(random.random()))

# Log and link the model to the Model Registry
run.link_model(path="./my_model.h5", registered_model_name="MNIST")

run.finish()
```

4. **モデルの移行を CI/CD ワークフロー に接続する**: 候補モデルを ワークフロー ステージを介して移行し、Webhooks で [ダウンストリーム アクション を自動化]({{< relref path="/guides/core/automations/" lang="ja" >}}) します。

## 開始方法
ユースケース に応じて、次の リソース を調べて W&B Models の使用を開始してください。

* 2 部構成のビデオ シリーズをご覧ください。
  1. [モデルの ログ 記録と登録](https://www.youtube.com/watch?si=MV7nc6v-pYwDyS-3&v=ZYipBwBeSKE&feature=youtu.be)
  2. Model Registry での [モデルの消費とダウンストリーム プロセス の自動化](https://www.youtube.com/watch?v=8PFCrDSeHzw)。
* W&B Python SDK コマンド のステップごとの概要については、[モデルのウォークスルー]({{< relref path="./walkthrough.md" lang="ja" >}}) を読んで、データセット Artifacts の作成、追跡、および使用に使用できます。
* 以下について学びます。
   * [保護されたモデルと アクセス 制御]({{< relref path="./access_controls.md" lang="ja" >}})。
   * [Registry を CI/CD プロセス に接続する方法]({{< relref path="/guides/core/automations/" lang="ja" >}})。
   * 新しいモデル バージョン が登録済みモデルにリンクされたときの [Slack 通知]({{< relref path="./notifications.md" lang="ja" >}}) を設定します。
* Model Registry が ML ワークフロー にどのように適合し、モデル管理にそれを使用する利点については、[こちら](https://wandb.ai/wandb_fc/model-registry-reports/reports/What-is-an-ML-Model-Registry---Vmlldzo1MTE5MjYx) の レポート を確認してください。
* W&B [Enterprise Model Management](https://www.wandb.courses/courses/enterprise-model-management) コース を受講して、以下を学びます。
  * W&B Model Registry を使用して、モデルの管理と バージョン 管理、 リネージ の追跡、およびさまざまなライフサイクル ステージでのモデルのプロモーションを行います。
  * Webhooks を使用して、モデル管理 ワークフロー を自動化します。
  * Model Registry が、モデルの 評価、モニタリング 、および デプロイメント のためのモデル開発ライフサイクルにおいて、外部 ML システム および ツール とどのように統合されるかを確認します。
