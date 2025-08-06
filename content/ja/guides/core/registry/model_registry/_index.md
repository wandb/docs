---
title: モデルレジストリ
description: トレーニングからプロダクションまでのモデルライフサイクルを管理するためのモデルレジストリ
menu:
  default:
    identifier: model-registry
    parent: registry
weight: 9
url: guides/core/registry/model_registry
cascade:
- url: guides/core/registry/model_registry/:filename
---

{{% alert %}}
W&B は将来的に W&B Model Registry のサポートを終了する予定です。ユーザーの皆さまには、モデルアーティファクトのバージョンを連携・共有するために [W&B Registry]({{< relref "/guides/core/registry/" >}}) のご利用をおすすめします。W&B Registry は従来の W&B Model Registry よりも幅広い機能を提供しています。詳細については [Registry のドキュメント]({{< relref "/guides/core/registry/" >}}) をご参照ください。

また、既存のモデルアーティファクトも、近い将来に従来の Model Registry から新しい W&B Registry へ移行されます。移行プロセスの詳細については [レガシー Model Registry からの移行]({{< relref "/guides/core/registry/model_registry_eol.md" >}}) をご覧ください。
{{% /alert %}}

W&B Model Registry は、チームで学習済みモデルを管理し、ML 実践者がプロダクション用候補を公開し、 downstream チームやステークホルダーが利用できるようにする仕組みです。ステージング/候補モデルを保管し、ステージングに関連したワークフローを管理するために利用します。

{{< img src="/images/models/model_reg_landing_page.png" alt="Model Registry" >}}

W&B Model Registry でできること：

* [各機械学習タスクごとに最良のモデルバージョンをブックマーク]({{< relref "./link-model-version.md" >}})
* [Automate]({{< relref "/guides/core/automations/" >}}) を使って downstream のプロセスやモデル CI/CD を自動化
* モデルバージョンを ML ライフサイクルの各段階（ステージングからプロダクションへ）に移動
* モデルのリネージを追跡し、プロダクションモデルの変更履歴を監査

{{< img src="/images/models/models_landing_page.png" alt="Models overview" >}}

## 仕組み
ステージングしたモデルは、いくつかのシンプルなステップでトラッキング・管理できます。

1. **モデルバージョンをログ**: トレーニングスクリプトで数行追加し、モデルファイルを W&B へアーティファクトとして保存します。
2. **パフォーマンスを比較**: ライブチャートでメトリクスやモデルの予測サンプルを比較し、どのモデルバージョンが最も良いか特定します。
3. **Registry へのリンク**: 最良のモデルバージョンを Registered Model にリンクしてブックマークします。これは Python でプログラム的にも、W&B の UI からも可能です。

以下のコードスニペットは、モデルを Model Registry にログしリンクする方法を示しています。

```python
import wandb
import random

# 新しい W&B run を開始
run = wandb.init(project="models_quickstart")

# モデルのメトリクスをシミュレートしてログ
run.log({"acc": random.random()})

# シミュレート用のモデルファイルを作成
with open("my_model.h5", "w") as f:
    f.write("Model: " + str(random.random()))

# モデルを Model Registry にログ & リンク
run.link_model(path="./my_model.h5", registered_model_name="MNIST")

run.finish()
```

4. **モデルの遷移を CI/CD ワークフローに接続**: 候補モデルをワークフローステージごとに移動し、[downstream のアクションを自動化]({{< relref "/guides/core/automations/" >}}) できます（webhook を活用）。

## 開始方法
目的に合わせて、以下のリソースを活用して W&B Models を始めましょう。

* 2 部構成の動画シリーズをご覧ください：
  1. [モデルのロギングと登録](https://www.youtube.com/watch?si=MV7nc6v-pYwDyS-3&v=ZYipBwBeSKE&feature=youtu.be)
  2. [モデルの利用と downstream プロセスの自動化](https://www.youtube.com/watch?v=8PFCrDSeHzw) （Model Registry 内）
* W&B Python SDK コマンドを使ったモデル作成・トラッキング・データセットアーティファクトの利用まで、[models walkthrough]({{< relref "./walkthrough.md" >}}) で順を追って確認できます。
* 次もチェックしましょう：
   * [Protected models（保護モデル）とアクセス制御について]({{< relref "./access_controls.md" >}})
   * [Registry を CI/CD プロセスに接続する方法]({{< relref "/guides/core/automations/" >}})
   * 新しいモデルバージョンを Registered Model にリンクした際に [Slack 通知]({{< relref "./notifications.md" >}}) を設定
* [What is an ML Model Registry?](https://wandb.ai/wandb_fc/model-registry-reports/reports/What-is-an-ML-Model-Registry---Vmlldzo1MTE5MjYx) をチェックして、Model Registry を ML ワークフローに取り入れる方法を学びましょう。
* W&B の [Enterprise Model Management](https://www.wandb.courses/courses/enterprise-model-management) コースで次を学びます：
  * W&B Model Registry の活用によるモデルの管理・バージョニング・リネージトラッキング・ライフサイクルステージの昇格
  * webhook を使ったモデル管理ワークフローの自動化
  * モデル評価・モニタリング・デプロイメントなど、外部 ML システムやツールと Model Registry がどのように統合されるか