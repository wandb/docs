---
description: トレーニングからプロダクションまでのモデルライフサイクルを管理するモデルレジストリ
slug: /guides/model_registry
displayed_sidebar: default
---


# Model registry
W&B Model Registry は、チームの訓練済みモデルを収容する場所であり、ML プラクティショナーがプロダクション候補として下流のチームやステークホルダーに提供できます。ステージングされたモデルや候補モデルを収容し、それに関連するワークフローを管理するために使用されます。

![](/images/models/model_reg_landing_page.png)

W&B Model Registry では、以下のことができます:

* [各機械学習タスクにおける最良のモデルバージョンをブックマークします。](./link-model-version.md)
* [下流プロセスとモデルの CI/CD を自動化します。](./automation.md)
* モデルバージョンをステージングからプロダクションまでの ML ライフサイクルを通じて移動させます。
* モデルのリネージを追跡し、プロダクションモデルへの変更履歴を監査します。

![](/images/models/models_landing_page.png)

## 仕組み
ステージングされたモデルを簡単な手順で追跡および管理します。

1. **モデルバージョンをログ**: トレーニングスクリプトに数行のコードを追加し、モデルファイルを W&B にアーティファクトとして保存します。
2. **パフォーマンスを比較**: ライブチャートを確認して、モデルのトレーニングおよび検証からのメトリクスとサンプル予測を比較します。どのモデルバージョンが最も優れているか識別します。
3. **レジストリへのリンク**: 最良のモデルバージョンをブックマークし、Python プログラムまたは W&B の UI にて登録モデルにリンクします。

以下のコードスニペットは、モデルをモデルレジストリにログおよびリンクする方法を示しています:

```python showLineNumbers
import wandb
import random

# 新しい W&B run を開始
run = wandb.init(project="models_quickstart")

# モデルメトリクスをログするのをシミュレート
run.log({"acc": random.random()})

# シミュレートされたモデルファイルを作成
with open("my_model.h5", "w") as f:
    f.write("Model: " + str(random.random()))

# モデルをログしてモデルレジストリにリンク
run.link_model(path="./my_model.h5", registered_model_name="MNIST")

run.finish()
```

4. **モデルの遷移を CI/DC ワークフローに接続**: 候補モデルをワークフローステージを通じて遷移させ、[下流アクションを自動化](./automation.md) します。

## 開始方法
ユースケースに応じて、W&B Models を始めるための以下のリソースを探索してください:

* 2 部構成のビデオシリーズをご覧ください:
  1. [モデルのログと登録](https://www.youtube.com/watch?si=MV7nc6v-pYwDyS-3&v=ZYipBwBeSKE&feature=youtu.be)
  2. Model Registry での[モデルの使用と下流プロセスの自動化](https://www.youtube.com/watch?v=8PFCrDSeHzw)
* W&B Python SDK コマンドを使用してデータセットアーティファクトを作成、追跡、使用するための手順を説明する[モデルのウォークスルー](./walkthrough.md)を読んでください。
* 次のことについて学んでください:
  * [保護されたモデルとアクセス制御](./access_controls.md)
  * [モデルレジストリを CI/CD プロセスに接続する方法](./automation.md)
  * 新しいモデルバージョンが登録モデルにリンクされたときに[Slack 通知を設定](./notifications.md)する方法
* モデル管理のために Model Registry がどのように ML ワークフローに適合するか、およびその利点についての[こちら](https://wandb.ai/wandb_fc/model-registry-reports/reports/What-is-an-ML-Model-Registry---Vmlldzo1MTE5MjYx)のレポートを確認してください。
* W&B の [Enterprise Model Management コース](https://www.wandb.courses/courses/enterprise-model-management)を受講し、以下のことを学んでください:
  * W&B Model Registry を使用してモデルを管理およびバージョン管理し、リネージを追跡し、ライフサイクルの異なる段階でモデルを促進する
  * Webhooks や Launch ジョブを使用してモデル管理ワークフローを自動化する
  * モデル開発ライフサイクルにおけるモデルの評価、監視、およびデプロイメントのために Model Registry が外部の機械学習システムやツールとどのように統合されるかを見る