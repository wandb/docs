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

OpenAI GPT-3.5 または GPT-4 モデルのファインチューニングの メトリクス と 設定 を W&B に ログ します。W&B エコシステム を利用して、ファインチューニング の 実験 、 モデル 、 データセット を 追跡 し、同僚と 結果 を共有します。

{{% alert %}}
ファインチューニング できる モデル の リスト については、[OpenAI のドキュメント](https://platform.openai.com/docs/guides/fine-tuning/which-models-can-be-fine-tuned) を 参照 してください。
{{% /alert %}}

OpenAI でのファインチューニングのために W&B を 統合 する 方法 の 詳細 については、OpenAI ドキュメント の [Weights and Biases Integration](https://platform.openai.com/docs/guides/fine-tuning/weights-and-biases-integration) セクション を 参照 してください。

## OpenAI Python API の インストール または 更新

W&B OpenAI ファインチューニング 連携 は、OpenAI バージョン 1.0 以降 で 動作 します。[OpenAI Python API](https://pypi.org/project/openai/) ライブラリ の 最新 バージョン については、PyPI ドキュメント を 参照 してください。

OpenAI Python API を インストール するには、次 を 実行 します。
```python
pip install openai
```

OpenAI Python API が すでに インストール されている 場合 は、次 の コマンド で 更新 できます。
```python
pip install -U openai
```

## OpenAI ファインチューニング の 結果 の 同期

W&B を OpenAI の ファインチューニング API と 統合 して、ファインチューニング の メトリクス と 設定 を W&B に ログ します。これ を 行う には、`wandb.integration.openai.fine_tuning` モジュール の `WandbLogger` クラス を 使用 します。

```python
from wandb.integration.openai.fine_tuning import WandbLogger

# Finetuning logic (ファインチューニング の ロジック)

WandbLogger.sync(fine_tune_job_id=FINETUNE_JOB_ID)
```

{{< img src="/images/integrations/open_ai_auto_scan.png" alt="" >}}

### ファインチューン を 同期 する

スクリプト から 結果 を 同期 します

```python
from wandb.integration.openai.fine_tuning import WandbLogger

# one line command (ワンライン コマンド)
WandbLogger.sync()

# passing optional parameters (オプション の パラメータ を 渡す)
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

| 引数                  | 説明                                                                                                                                                                                                                                                                                 |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| fine_tune_job_id         | これは、`client.fine_tuning.jobs.create` を 使用 して ファインチューン ジョブ を 作成 する とき に 取得 する OpenAI Fine-Tune ID です。この 引数 が None (デフォルト) の 場合 、まだ W&B に 同期 されていない すべて の OpenAI ファインチューン ジョブ が W&B に 同期 されます。                                                                                                                                                                                                          |
| openai_client            | 初期化 された OpenAI クライアント を `sync` に 渡します。クライアント が 提供 されない 場合 、ロガー 自体 によって 初期化 されます。デフォルト では None です。 |
| num_fine_tunes           | ID が 提供 されない 場合 、同期 されていない すべて の ファインチューン が W&B に ログ されます。この 引数 を 使用 すると、同期 する 最近 の ファインチューン の 数 を 選択 できます。num_fine_tunes が 5 の 場合 、最新 の 5 つ の ファインチューン が 選択 されます。                                                                                                                   |
| project                  | ファインチューン の メトリクス 、 モデル 、 データ など が ログ される Weights and Biases プロジェクト 名。デフォルト では、プロジェクト 名 は "OpenAI-Fine-Tune" です。                                                                                                   |
| entity                   | run の 送信 先 と なる W&B ユーザー 名 または チーム 名。デフォルト では、デフォルト の エンティティ (通常 は ユーザー 名) が 使用 されます。                                                                                                                                                                          |
| overwrite                | 同じ ファインチューン ジョブ の 既存 の wandb run を 強制 的 に ログ および 上書き します。デフォルト では False です。                                                                                                                                                                                                         |
| wait_for_job_success     | OpenAI ファインチューニング ジョブ が 開始 される と、通常 は 少し 時間 が かかります。ファインチューン ジョブ が 完了 すると すぐ に メトリクス が W&B に ログ される よう に する ため に、この 設定 は 60 秒 ごと に ファインチューン ジョブ の ステータス が `succeeded` に 変わる か どう か を 確認 します。ファインチューン ジョブ が 成功 した と 検出 される と、メトリクス は 自動 的 に W&B に 同期 されます。デフォルト では True に 設定 されます。 |
| model_artifact_name      | ログ に 記録 される モデル Artifacts の 名前。デフォルト は `"model-metadata"` です。                    |
| model_artifact_type      | ログ に 記録 される モデル Artifacts の タイプ。デフォルト は `"model"` です。                    |
| \*\*kwargs_wandb_init  | [`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ja" >}}) に 直接 渡される 追加 の 引数                    |

## データセット の バージョン管理 と 可視化

### バージョン管理

ファインチューニング の ため に OpenAI に アップロード する トレーニング データ と 検証 データ は、より 簡単 な バージョン管理 の ため に W&B Artifacts として 自動 的 に ログ に 記録 されます。以下 は、Artifacts 内 の トレーニング ファイル の ビュー です。ここでは、この ファイル を ログ に 記録 した W&B run、ログ に 記録 された 時刻、これ が データセット の どの バージョン で ある か、メタデータ、および トレーニング データ から トレーニング 済み モデル への DAG リネージ を 確認 できます。

{{< img src="/images/integrations/openai_data_artifacts.png" alt="" >}}

### 可視化

データセット は W&B Tables として 可視化 され、データセット の 探索、検索、および 操作 が 可能 に なります。以下 に示す W&B Tables を 使用 して 可視化 された トレーニング サンプル を ご覧 ください。

{{< img src="/images/integrations/openai_data_visualization.png" alt="" >}}

## ファインチューニング された モデル と モデル の バージョン管理

OpenAI は ファインチューニング された モデル の ID を 提供 します。モデル の 重み に アクセス できない ため、`WandbLogger` は モデル の すべて の 詳細 (ハイパー パラメータ 、 データ ファイル ID など) と `fine_tuned_model` ID を 含む `model_metadata.json` ファイル を 作成 し、W&B Artifacts として ログ に 記録 します。

この モデル (メタデータ) Artifacts は、[W&B Model Registry]({{< relref path="/guides/models/registry/model_registry/" lang="ja" >}}) 内 の モデル に さらに リンク できます。

{{< img src="/images/integrations/openai_model_metadata.png" alt="" >}}

## よく ある 質問

### W&B で チーム と ファインチューン の 結果 を 共有 するには どうすれば よい ですか?

次 を 使用 して、チーム アカウント に ファインチューン ジョブ を ログ します。

```python
WandbLogger.sync(entity="YOUR_TEAM_NAME")
```

### run を 整理 するには どうすれば よい ですか?

W&B run は 自動 的 に 整理 され、ジョブタイプ 、 ベース モデル 、 学習 レート 、 トレーニング ファイル 名 、 その他 の ハイパーパラメータ など、任意 の 設定 パラメータ に 基づい て フィルタリング/ソート できます。

さらに、run の 名前 を 変更 したり、メモ を 追加 したり、タグ を 作成 して グループ化 したり できます。

満足 したら、ワークスペース を 保存 し、それ を 使用 して レポート を 作成 し、run と 保存 された Artifacts (トレーニング/検証 ファイル) から データ を インポート できます。

### ファインチューニング された モデル に アクセス するには どうすれば よい ですか?

ファインチューニング された モデル ID は、Artifacts (`model_metadata.json`) として W&B に ログ される とともに、設定 としても ログ されます。

```python
import wandb

ft_artifact = wandb.run.use_artifact("ENTITY/PROJECT/model_metadata:VERSION")
artifact_dir = artifact.download()
```

ここで、`VERSION` は 次 の いずれか です。

* `v2` など の バージョン番号
* `ft-xxxxxxxxx` など の ファインチューン ID
* `latest` など、自動 的 に 追加 される エイリアス または 手動 で 追加 される エイリアス

次 に、ダウンロード した `model_metadata.json` ファイル を 読み取る こと で `fine_tuned_model` ID に アクセス できます。

### ファインチューン が 正常 に 同期 されなかった 場合 は どう なり ますか?

ファインチューン が W&B に 正常 に ログ されなかった 場合 は、`overwrite=True` を 使用 して ファインチューン ジョブ ID を 渡す ことができます。

```python
WandbLogger.sync(
    fine_tune_job_id="FINE_TUNE_JOB_ID",
    overwrite=True,
)
```

### W&B で データセット と モデル を 追跡 できますか?

トレーニング データ と 検証 データ は、Artifacts として W&B に 自動 的 に ログ されます。ファインチューニング された モデル の ID を 含む メタデータ も Artifacts として ログ されます。

`wandb.Artifact`、`wandb.log` など の 低 レベル の wandb API を 使用 して、パイプライン を 常に 制御 できます。これにより、 データ と モデル の 完全 な トレーサビリティ が 実現 します。

{{< img src="/images/integrations/open_ai_faq_can_track.png" alt="" >}}

## 参考資料

* [OpenAI ファインチューニング ドキュメント](https://platform.openai.com/docs/guides/fine-tuning/) は 非常 に 徹底 的 で、多く の 役立つ ヒント が 含ま れています
* [デモ Colab](http://wandb.me/openai-colab)
* [W&B で OpenAI GPT-3.5 および GPT-4 モデル を ファインチューニング する 方法](http://wandb.me/openai-report) レポート
