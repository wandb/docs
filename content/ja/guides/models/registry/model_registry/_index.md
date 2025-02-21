---
title: Model registry
description: '**モデルレジストリ** は、トレーニングからプロダクションまでの モデル のライフサイクルを管理します。'
cascade:
- url: guides/model_registry/:filename
menu:
  default:
    identifier: ja-guides-models-registry-model_registry-_index
    parent: registry
url: guides/model_registry
weight: 9
---

{{% alert %}}
W&B は、2024 年以降 W&B Model Registry をサポートしなくなります。代わりに、[W&B Registry]({{< relref path="/guides/models/registry/" lang="ja" >}}) を使用して、model artifacts バージョンをリンクおよび共有することをお勧めします。W&B Registry は、従来の W&B Model Registry の機能を拡張します。W&B Registry の詳細については、[Registry のドキュメント]({{< relref path="/guides/models/registry/" lang="ja" >}}) を参照してください。

W&B は、従来の Model Registry にリンクされている既存の model artifacts を、2024 年の秋または初冬に新しい W&B Registry に移行します。移行プロセスについては、[従来の Model Registry からの移行]({{< relref path="/guides/models/registry/model_registry_eol.md" lang="ja" >}}) を参照してください。
{{% /alert %}}

W&B Model Registry は、チームでトレーニングされたモデルを保管する場所で、ML 実務者は、ダウンストリームのチームやステークホルダーが利用するために、プロダクションの候補を公開できます。これは、ステージングされたモデルや候補モデルを保管し、ステージングに関連するワークフローを管理するために使用されます。

{{< img src="/images/models/model_reg_landing_page.png" alt="" >}}

W&B Model Registry を使用すると、次のことが可能になります。

* [各 機械学習 タスクに最適なモデル バージョンをブックマークする。]({{< relref path="./link-model-version.md" lang="ja" >}})
* ダウンストリームのプロセスとモデル CI/CD を [自動化]({{< relref path="/guides/models/automations/model-registry-automations.md" lang="ja" >}}) する。
* モデル バージョンを ML ライフサイクル (ステージングからプロダクション) に移行する。
* モデル の リネージ を追跡し、プロダクション モデル への変更履歴を監査する。

{{< img src="/images/models/models_landing_page.png" alt="" >}}

## 仕組み
いくつかの簡単なステップで、ステージングされたモデルを追跡および管理します。

1. **モデル バージョンを ログ に記録する**: トレーニング スクリプト で、数行の コード を追加して、モデル ファイル を Artifacts として W&B に保存します。
2. **パフォーマンスを比較する**: ライブ チャート を確認して、モデル トレーニング と バリデーション からの メトリクス とサンプル 予測 を比較します。どのモデル バージョン が最もパフォーマンスが高いかを特定します。
3. **Registry にリンクする**: Python でプログラム的に、または W&B UI でインタラクティブに、登録されたモデルにリンクして、最適なモデル バージョン をブックマークします。

次の コード スニペット は、モデル を Model Registry に ログ してリンクする方法を示しています。

```python showLineNumbers
import wandb
import random

# 新しい W&B run を開始します
run = wandb.init(project="models_quickstart")

# モデル の メトリクス の ログ をシミュレートします
run.log({"acc": random.random()})

# シミュレートされたモデル ファイル を作成します
with open("my_model.h5", "w") as f:
    f.write("Model: " + str(random.random()))

# モデル を ログ に記録し、Model Registry にリンクします
run.link_model(path="./my_model.h5", registered_model_name="MNIST")

run.finish()
```

4. **モデル の移行を CI/DC ワークフロー に接続する**: 候補モデル をワークフロー の段階を経て移行し、webhook または job で [ダウンストリーム のアクション を自動化]({{< relref path="/guides/models/automations/model-registry-automations.md" lang="ja" >}}) します。

## 開始方法
ユースケース に応じて、次のリソース を調べて W&B Models を開始してください。

* 2 部構成のビデオ シリーズ をご覧ください。
  1. [モデル の ログ と登録](https://www.youtube.com/watch?si=MV7nc6v-pYwDyS-3&v=ZYipBwBeSKE&feature=youtu.be)
  2. Model Registry での [モデル の消費とダウンストリーム プロセス の自動化](https://www.youtube.com/watch?v=8PFCrDSeHzw)。
* W&B Python SDK コマンド のステップバイステップ の概要については、[モデル のウォークスルー]({{< relref path="./walkthrough.md" lang="ja" >}}) を参照して、データセット Artifacts の作成、追跡、および使用に使用できます。
* 以下について学びます。
   * [保護されたモデル と アクセス 制御]({{< relref path="./access_controls.md" lang="ja" >}})。
   * [Model Registry を CI/CD プロセス に接続する方法]({{< relref path="/guides/models/automations/model-registry-automations.md" lang="ja" >}})。
   * 新しいモデル バージョン が登録されたモデル にリンクされている場合、[Slack 通知]({{< relref path="./notifications.md" lang="ja" >}}) を設定します。
* Model Registry が ML ワークフロー にどのように適合し、モデル 管理 に使用するメリット については、[こちら](https://wandb.ai/wandb_fc/model-registry-reports/reports/What-is-an-ML-Model-Registry---Vmlldzo1MTE5MjYx) の レポート を確認してください。
* W&B [Enterprise Model Management](https://www.wandb.courses/courses/enterprise-model-management) コース を受講して、以下を学びます。
  * W&B Model Registry を使用して、モデル の管理と バージョン 管理、リネージ の追跡、およびさまざまな ライフサイクル 段階でのモデル の昇格を行います。
  * webhook を使用して、モデル 管理 ワークフロー を自動化します。
  * Model Registry が、モデル の 評価 、監視、および デプロイメント のためのモデル 開発 ライフサイクル における外部 ML システム および ツール とどのように統合されるかを確認します。
