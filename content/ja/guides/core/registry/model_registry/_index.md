---
title: モデルレジストリ
description: モデルレジストリで、トレーニングからプロダクションまでのモデルのライフサイクルを管理
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
W&B は今後、W&B Model Registry のサポートを終了する予定です。ユーザーのみなさまは、モデルアーティファクトのバージョン管理や共有には [W&B Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) の利用をおすすめします。W&B Registry は旧 W&B Model Registry の機能を拡張しています。詳細は [Registry ドキュメント]({{< relref path="/guides/core/registry/" lang="ja" >}}) をご覧ください。

W&B は、既存の旧 Model Registry に紐づくモデルアーティファクトを、近い将来、新しい W&B Registry へ移行します。移行プロセスについては [旧 Model Registry からの移行]({{< relref path="/guides/core/registry/model_registry_eol.md" lang="ja" >}}) をご参照ください。
{{% /alert %}}

W&B Model Registry は、チームでトレーニングしたモデルを管理する場所です。ML 実務者がプロダクション用候補モデルを公開し、下流のチームや関係者が利用できるようにします。候補モデルの管理やステージングにまつわるワークフローの運用に活用されます。

{{< img src="/images/models/model_reg_landing_page.png" alt="Model Registry" >}}

W&B Model Registry では、次のことができます：

* [各機械学習タスクでベストなモデルバージョンをブックマークできます。]({{< relref path="./link-model-version.md" lang="ja" >}})
* [オートメーション]({{< relref path="/guides/core/automations/" lang="ja" >}})を活用して下流プロセスやモデルの CI/CD を自動化できます。
* モデルバージョンをステージングからプロダクションまで ML ライフサイクルに沿って管理できます。
* モデルのリネージや、プロダクションモデルへの変更履歴の監査ができます。

{{< img src="/images/models/models_landing_page.png" alt="Models overview" >}}

## 仕組み
数ステップでステージ済みモデルのトラッキングと管理ができます。

1. **モデルバージョンをログする**: トレーニングスクリプトに数行追加するだけで、モデルファイルを W&B のアーティファクトとして保存できます。
2. **パフォーマンスを比較する**: ライブチャートで、モデルのトレーニングや検証によるメトリクスやサンプル予測を比較できます。最良のモデルバージョンを特定しましょう。
3. **Registry へリンクする**: Python でプログラムから、または W&B UI でインタラクティブに、最良のモデルバージョンを registered model にリンクしてブックマークします。

次のコードスニペットは、モデルを Model Registry へログしリンクする例です：

```python
import wandb
import random

# 新しい W&B run を開始
run = wandb.init(project="models_quickstart")

# モデルメトリクスをシミュレートしてログ
run.log({"acc": random.random()})

# モデルファイルを作成する（サンプル）
with open("my_model.h5", "w") as f:
    f.write("Model: " + str(random.random()))

# モデルを Model Registry にログし、リンクする
run.link_model(path="./my_model.h5", registered_model_name="MNIST")

run.finish()
```

4. **モデルのステージ遷移を CI/CD ワークフローと連携**: 候補モデルのステージをワークフローで進めて、[下流のアクションを自動化]({{< relref path="/guides/core/automations/" lang="ja" >}})するために webhook を活用します。


## 開始方法
ユースケースごとに、次のリソースを活用して W&B Models を始めましょう：

* 2部構成の動画シリーズもご覧ください：
  1. [モデルのログ・登録方法](https://www.youtube.com/watch?si=MV7nc6v-pYwDyS-3&v=ZYipBwBeSKE&feature=youtu.be)
  2. [Model Registry でのモデル活用・下流プロセス自動化](https://www.youtube.com/watch?v=8PFCrDSeHzw)
* [models ウォークスルー]({{< relref path="./walkthrough.md" lang="ja" >}}) で、W&B Python SDK コマンドによるデータセットアーティファクトの作成・トラッキング・利用方法をステップごとに確認できます。
* 次の内容もチェックしましょう:
   * [保護されたモデルとアクセス管理]({{< relref path="./access_controls.md" lang="ja" >}})
   * [Registry と CI/CD プロセスの連携方法]({{< relref path="/guides/core/automations/" lang="ja" >}})
   * 新しいモデルバージョンが registered model にリンクされた際に [Slack 通知]({{< relref path="./notifications.md" lang="ja" >}}) を設定する方法
* [What is an ML Model Registry?](https://wandb.ai/wandb_fc/model-registry-reports/reports/What-is-an-ML-Model-Registry---Vmlldzo1MTE5MjYx) を読んで、Model Registry を ML ワークフローにどう組み込むか学びましょう。
* W&B の [Enterprise Model Management](https://www.wandb.courses/courses/enterprise-model-management) コースもおすすめです。以下が学べます：
  * W&B Model Registry を使ったモデルの管理・バージョン管理、リネージのトラッキング、モデルのライフサイクル管理
  * webhook を用いたモデル管理ワークフローの自動化
  * Model Registry が外部の ML システムやツールとどのように連携し、モデルの評価・監視・デプロイメントに活用できるか
