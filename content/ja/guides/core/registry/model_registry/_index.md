---
title: モデルレジストリ
description: モデルレジストリでトレーニングからプロダクションまでのモデルライフサイクルを管理する
cascade:
- url: /ja/guides/core/registry/model_registry/:filename
menu:
  default:
    identifier: ja-guides-core-registry-model_registry-_index
    parent: registry
url: /ja/guides/core/registry/model_registry
weight: 9
---

{{% alert %}}
W&B は最終的に W&B Model Registry のサポートを停止します。ユーザーは代わりにモデルのアーティファクトバージョンをリンクして共有するために [W&B Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) を使用することを推奨されます。W&B Registry は、旧 W&B Model Registry の機能を拡張します。W&B Registry について詳しくは、[Registry docs]({{< relref path="/guides/core/registry/" lang="ja" >}}) をご覧ください。

W&B は近い将来、旧 Model Registry にリンクされた既存のモデルアーティファクトを新しい W&B Registry へ移行する予定です。移行プロセスに関する情報は、[Migrating from legacy Model Registry]({{< relref path="/guides/core/registry/model_registry_eol.md" lang="ja" >}}) をご覧ください。
{{% /alert %}}

W&B Model Registry は、チームのトレーニングされたモデルを収納し、MLプラクティショナーがプロダクションに向けた候補を公開し、下流のチームや関係者に消費させることができます。これは、ステージング/候補モデルを収容し、ステージングに関連するワークフローを管理するために使用されます。

{{< img src="/images/models/model_reg_landing_page.png" alt="" >}}

W&B Model Registry を使用すると、以下が可能です：

* [機械学習タスクごとにベストなバージョンのモデルをブックマークする。]({{< relref path="./link-model-version.md" lang="ja" >}})
* 下流のプロセスとモデル CI/CD を [オートメーション化する]({{< relref path="/guides/core/automations/" lang="ja" >}})。
* モデルバージョンをステージングからプロダクションまで ML ライフサイクルを通して移行する。
* モデルのリネージを追跡し、プロダクションモデルの変更履歴を監査する。

{{< img src="/images/models/models_landing_page.png" alt="" >}}

## 仕組み
ステージングされたモデルを数ステップで追跡し、管理します。

1. **モデルバージョンをログする**：トレーニングスクリプトに数行のコードを追加して、モデルファイルをアーティファクトとして W&B に保存します。
2. **パフォーマンスを比較する**：ライブチャートをチェックして、トレーニングと検証からのメトリクスやサンプル予測を比較します。どのモデルバージョンが最もよくパフォーマンスしたかを特定します。
3. **レジストリにリンクする**：ベストなモデルバージョンを登録済みモデルにリンクしてブックマークします。これは Python でプログラム的に、または W&B UI でインタラクティブに行うことができます。

以下のコードスニペットは、モデルを Model Registry にログし、リンクする方法を示しています：

```python
import wandb
import random

# 新しい W&B run を開始
run = wandb.init(project="models_quickstart")

# モデルメトリクスをシミュレーションしてログする
run.log({"acc": random.random()})

# シミュレートされたモデルファイルを作成
with open("my_model.h5", "w") as f:
    f.write("Model: " + str(random.random()))

# モデルを Model Registry にログし、リンクする
run.link_model(path="./my_model.h5", registered_model_name="MNIST")

run.finish()
```

4. **モデルの移行を CI/CD ワークフローに接続する**：候補モデルをワークフローステージを通して移行し、[下流のアクションをオートメーション化する]({{< relref path="/guides/core/automations/" lang="ja" >}})ことを Webhook を使って行います。

## 開始方法
ユースケースに応じて、W&B Models を使い始めるための以下のリソースを探ります。

* 2 部構成のビデオシリーズを確認：
  1. [モデルのログと登録](https://www.youtube.com/watch?si=MV7nc6v-pYwDyS-3&v=ZYipBwBeSKE&feature=youtu.be)
  2. [モデルの消費と下流プロセスのオートメーション化](https://www.youtube.com/watch?v=8PFCrDSeHzw) in the Model Registry.
* [モデルウォークスルー]({{< relref path="./walkthrough.md" lang="ja" >}})を読み、W&B Python SDK コマンドを使用してデータセットアーティファクトを作成、追跡、および使用する手順を確認します。
* 以下について学ぶ：
  * [保護されたモデルとアクセス制御]({{< relref path="./access_controls.md" lang="ja" >}})。
  * [CI/CD プロセスにレジストリを接続する方法]({{< relref path="/guides/core/automations/" lang="ja" >}})。
  * 新しいモデルバージョンが登録済みモデルにリンクされたときの [Slack 通知を設定]({{< relref path="./notifications.md" lang="ja" >}})。
* Model Registry があなたの ML ワークフローにどのようにフィットし、モデル管理のためにそれを使用することの利点についての [この](https://wandb.ai/wandb_fc/model-registry-reports/reports/What-is-an-ML-Model-Registry---Vmlldzo1MTE5MjYx) レポートを確認します。
* W&B の [Enterprise Model Management](https://www.wandb.courses/courses/enterprise-model-management) コースを受講し、以下を学びます：
  * W&B Model Registry を使って、モデルを管理、バージョン化し、リネージを追跡し、様々なライフサイクルステージを通じてモデルを推進する方法。
  * Webhook を使ってモデル管理ワークフローをオートメーション化する方法。
  * モデル評価、監視、デプロイメントのために Model Registry が外部 ML システムやツールとどのように統合されているかを確認する。