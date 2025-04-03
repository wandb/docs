---
title: OpenAI Fine-Tuning
description: W&B を使用して OpenAI モデルを ファインチューン する方法。
menu:
  default:
    identifier: ja-guides-integrations-openai-fine-tuning
    parent: integrations
weight: 250
---

{{< cta-button colabLink="http://wandb.me/openai-colab" >}}

OpenAI GPT-3.5 または GPT-4 モデルのファインチューニングのメトリクスと設定を W&B に記録します。W&B の エコシステム を利用して、ファインチューニング の 実験 、 モデル 、 データセット を追跡し、同僚と 結果 を共有します。

{{% alert %}}
ファインチューニング できる モデル のリストについては、[OpenAI のドキュメント](https://platform.openai.com/docs/guides/fine-tuning/which-models-can-be-fine-tuned)を参照してください。
{{% /alert %}}

W&B と OpenAI を ファインチューニング 用に 統合 する方法に関する補足情報については、OpenAI のドキュメントの[Weights and Biases Integration](https://platform.openai.com/docs/guides/fine-tuning/weights-and-biases-integration)セクションを参照してください。

## OpenAI Python API のインストールまたはアップデート

W&B OpenAI ファインチューニング インテグレーション は、OpenAI バージョン 1.0 以降で動作します。[OpenAI Python API](https://pypi.org/project/openai/) ライブラリの最新バージョンについては、PyPI のドキュメントを参照してください。

OpenAI Python API をインストールするには、以下を実行します。
```python
pip install openai
```

OpenAI Python API がすでにインストールされている場合は、以下を実行してアップデートできます。
```python
pip install -U openai
```

## OpenAI ファインチューニング の 結果 を 同期 する

W&B を OpenAI の ファインチューニング API と 統合 して、ファインチューニング の メトリクス と 設定 を W&B に 記録 します。これを行うには、`wandb.integration.openai.fine_tuning` モジュールの `WandbLogger` クラスを使用します。

```python
from wandb.integration.openai.fine_tuning import WandbLogger

# Finetuning logic

WandbLogger.sync(fine_tune_job_id=FINETUNE_JOB_ID)
```

{{< img src="/images/integrations/open_ai_auto_scan.png" alt="" >}}

### ファインチューン を 同期 する

スクリプト から 結果 を 同期 します。

```python
from wandb.integration.openai.fine_tuning import WandbLogger

# one line command
WandbLogger.sync()

# passing optional parameters
WandbLogger.sync(
    fine_tune_job_id=None,
    num_fine_tunes=None,
    project="OpenAI-Fine-Tune",
    entity=None,
    overwrite=False,
    model_artifact_name="model-metadata",
    model_artifact_type="model",
    **kwargs_wandb_init
)
```

### リファレンス

| 引数                   | 説明                                                                                                                                                   |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| fine_tune_job_id         | これは、`client.fine_tuning.jobs.create` を使用して ファインチューン ジョブ を 作成 するときに取得する OpenAI Fine-Tune ID です。この 引数 が None (デフォルト) の場合、まだ W&B に 同期 されていないすべての OpenAI ファインチューン ジョブ が W&B に 同期 されます。                                                                                                      |
| openai_client            | 初期化された OpenAI クライアント を `sync` に渡します。クライアント が 提供 されない場合、ロガー自体によって初期化されます。デフォルトでは None です。                                                             |
| num_fine_tunes           | ID が 提供 されない場合、同期 されていないすべての ファインチューン が W&B に 記録 されます。この 引数 を使用すると、 同期 する 最新 の ファインチューン の 数 を 選択 できます。num_fine_tunes が 5 の場合、最新 の 5 つの ファインチューン が 選択 されます。                                                                                              |
| project                  | ファインチューン の メトリクス 、 モデル 、 データ などが 記録 される Weights and Biases プロジェクト 名。デフォルトでは、 プロジェクト 名は "OpenAI-Fine-Tune" です。                                                                |
| entity                   | run の送信先の W&B ユーザー 名または チーム 名。デフォルトでは、デフォルト の エンティティ が使用されます。通常は ユーザー 名です。                                                                                                   |
| overwrite                | 同じ ファインチューン ジョブ の 既存 の wandb run を 強制的に ログ に 記録 して 上書き します。デフォルトでは False です。                                                                                                         |
| wait_for_job_success     | OpenAI の ファインチューニング ジョブ が 開始 されると、通常、少し時間がかかります。メトリクス が ファインチューン ジョブ の 完了後すぐに W&B に 記録 されるようにするために、この 設定 では、60 秒ごとに ファインチューン ジョブ の ステータス が `succeeded` に 変わるかどうかを チェック します。ファインチューン ジョブ が 成功 したと 検出 されると、メトリクス は 自動的に W&B に 同期 されます。デフォルトでは True に 設定 されています。                                                                                 |
| model_artifact_name      | ログ に 記録 される モデル Artifacts の 名前。デフォルトは `"model-metadata"` です。                                                                                    |
| model_artifact_type      | ログ に 記録 される モデル Artifacts の タイプ。デフォルトは `"model"` です。                                                                                      |
| \*\*kwargs_wandb_init  | [`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ja" >}}) に 直接 渡される 追加 の 引数 。                                                                                                    |

## データセット の バージョン管理 と 可視化

### バージョン管理

ファインチューニング 用に OpenAI に アップロード する トレーニング データ と 検証 データ は、より簡単な バージョン 管理のために W&B Artifacts として 自動的に ログ に 記録 されます。以下は、Artifacts の トレーニング ファイル の ビュー です。ここでは、この ファイル を ログ に 記録 した W&B run、ログ に 記録 された日時、これが データセット のどの バージョン であるか、メタデータ 、および トレーニングデータ から トレーニング された モデル への DAG リネージ を確認できます。

{{< img src="/images/integrations/openai_data_artifacts.png" alt="" >}}

### 可視化

データセット は W&B テーブル として 可視化 され、データセット の 探索 、 検索 、および 操作 を行うことができます。以下の W&B テーブル を使用して 可視化 された トレーニング サンプル を チェック してください。

{{< img src="/images/integrations/openai_data_visualization.png" alt="" >}}

## ファインチューニング された モデル と モデル の バージョン管理

OpenAI は、ファインチューニング された モデル の ID を 提供 します。モデル の 重み に アクセス できないため、`WandbLogger` は、 モデル の すべての 詳細 ( ハイパーパラメーター 、 データ ファイル ID など) と `fine_tuned_model` ID を含む `model_metadata.json` ファイル を 作成 し、W&B Artifacts として ログ に 記録 します。

この モデル ( メタデータ ) Artifacts は、[W&B Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) の モデル にさらに リンク できます。

{{< img src="/images/integrations/openai_model_metadata.png" alt="" >}}

## よくある質問

### チーム で ファインチューン の 結果 を W&B で共有するにはどうすればよいですか?

以下を使用して、ファインチューン ジョブ を チーム アカウント に ログ に 記録 します。

```python
WandbLogger.sync(entity="YOUR_TEAM_NAME")
```

### run を 整理 するにはどうすればよいですか?

W&B run は 自動的に 整理 され、 ジョブタイプ 、 ベース モデル 、 学習率 、 トレーニング ファイル名 、その他の ハイパーパラメーター など、任意の設定 パラメータ に 基づいて フィルタリング/ソート できます。

さらに、run の 名前 を 変更 したり、メモ を 追加 したり、 タグ を 作成 して グループ化 したりできます。

満足したら、 ワークスペース を 保存 し、それを使用して レポート を 作成 し、run と 保存 された Artifacts ( トレーニング / 検証 ファイル ) から データ を インポート できます。

### ファインチューニング された モデル に アクセス するにはどうすればよいですか?

ファインチューニング された モデル ID は、Artifacts (`model_metadata.json`) および 設定 として W&B に ログ に 記録 されます。

```python
import wandb

ft_artifact = wandb.run.use_artifact("ENTITY/PROJECT/model_metadata:VERSION")
artifact_dir = artifact.download()
```

ここで、`VERSION` は次のいずれかです。

* `v2` などの バージョン 番号
* `ft-xxxxxxxxx` などの ファインチューン ID
* `latest` や 手動 で 追加 された エイリアス など、自動的に 追加 された エイリアス

次に、ダウンロード した `model_metadata.json` ファイル を 読み取ることで、`fine_tuned_model` ID に アクセス できます。

### ファインチューン が 正常 に 同期 されなかった場合はどうなりますか?

ファインチューン が W&B に 正常 に ログ に 記録 されなかった場合は、`overwrite=True` を使用して、ファインチューン ジョブ ID を 渡すことができます。

```python
WandbLogger.sync(
    fine_tune_job_id="FINE_TUNE_JOB_ID",
    overwrite=True,
)
```

### W&B で データセット と モデル を 追跡 できますか?

トレーニング および 検証 データ は、Artifacts として W&B に 自動的に ログ に 記録 されます。ファインチューニング された モデル の ID を含む メタデータ も、Artifacts として ログ に 記録 されます。

`wandb.Artifact`、`wandb.log` などの 低レベル の wandb API を使用して パイプライン を 常に 制御 できます。これにより、 データ と モデル の 完全 な トレーサビリティ が 可能 になります。

{{< img src="/images/integrations/open_ai_faq_can_track.png" alt="" >}}

## リソース

* [OpenAI Fine-tuning Documentation](https://platform.openai.com/docs/guides/fine-tuning/) は非常に詳細で、多くの役立つ ヒント が含まれています。
* [デモ Colab](http://wandb.me/openai-colab)
* [How to Fine-Tune Your OpenAI GPT-3.5 and GPT-4 Models with W&B](http://wandb.me/openai-report) レポート
