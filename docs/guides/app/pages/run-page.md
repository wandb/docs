---
description: それぞれのモデルのトレーニング run は、より大きなプロジェクト内で整理され、専用のページが与えられます
displayed_sidebar: default
---


# Run Page
プロジェクト内の特定のrunに関する詳細情報を探索するには、runページを使用します。

## Overviewタブ
Overviewタブを使用して、プロジェクト内の特定のrunについて学びます。例えば以下のような情報です:

* **Name**: runの名前
* **Description**: runの説明
* **Author**: runを作成したW&Bエンティティ
* **State**: runの[状態](#run-states)
* **Start time**: runが初期化されたタイムスタンプ
* **Duration**: runが状態に達するまでにかかった時間
* **Run path**: ユニークなrunパス `<entity>/<project>/<run_id>`
* **OS**: runを初期化したオペレーティングシステム
* **Python version**: runを作成したPythonバージョン
* **Git repository**: [Gitが有効な場合](../settings-page/user-settings.md#personal-github-integration)、runに関連付けられたGitリポジトリ
* **Command**: runを初期化したコマンド
* **System hardware**: runが実行されたハードウェア
* **Config**: [`wandb.config`](../../../guides/track/config.md)と共に保存された設定パラメータのリスト
* **Summary**: [`wandb.log()`](../../../guides/track/log/intro.md)と共に保存された要約パラメータのリスト、デフォルトでは最後にログされた値に設定

runの概要ページを表示するには:
1. プロジェクトワークスペース内で、特定のrunをクリックします。
2. 次に、左側の列にある **Overview** タブをクリックします。

![W&B Dashboard run overview tab](/images/app_ui/wandb_run_overview_page.png)

### Run states
次の表に、runが取る可能性がある状態を示します:

| State | Description |
| ----- | ----- |
| Finished| runが終了し、データが完全に同期された、または `wandb.finish()` が呼び出された |
| Failed | runが非ゼロ終了ステータスと共に終了した |
| Crashed | 内部プロセスでrunがハートビートを送信するのを停止した。これはマシンがクラッシュした場合に発生する可能性があります |
| Running | runがまだ実行中で、最近ハートビートを送信した |

## Workspaceタブ
Workspaceタブを使用して、特定のrunに関連する以下のような可視化を表示、検索、グループ化、整理します:

* 検証セットによる予測の例などの自動生成されたプロット
* カスタムプロット
* システムメトリクス など

![](/images/app_ui/wandb-run-page-workspace-tab.png)

W&B App UIを使用して手動でチャートを作成するか、またはW&B Python SDKを使用してプログラム的に作成します。詳細については、[Log media and objects in Experiments](../../track/log/intro.md)を参照してください。

## Systemタブ
**Systemタブ**には、特定のrunに対して追跡されたシステムメトリクスが表示されます。例えば:

* CPU利用率の可視化
* システムメモリ
* ディスクI/O
* ネットワークトラフィック
* GPU利用率
* GPU温度
* GPUがメモリにアクセスした時間
* GPUメモリのアロケーション
* GPUの電力使用量

[ここでライブ例を表示](https://wandb.ai/stacey/deep-drive/runs/ki2biuqy/system?workspace=user-carey)。

![](/images/app_ui/wandb_system_utilization.png)

Lambda Labsによる["See the Tracking System Resource"](https://lambdalabs.com/blog/weights-and-bias-gpu-cpu-utilization/) ブログでは、W&Bシステムメトリクスの使用方法に関する詳細情報が提供されています。

## Modelタブ
**Modelタブ**で、モデルのレイヤー数、パラメータ数、および各レイヤーの出力形状を確認します。

[ここでライブ例を表示](https://app.wandb.ai/stacey/deep-drive/runs/pr0os44x/model)。

![](/images/app_ui/wandb_run_page_model_tab.png)

## Logsタブ
**Logsタブ**には、コマンドラインで出力されたstdoutやstderrなどの出力が表示されます。W&Bは最後の10,000行を表示します。

右上の **Download** ボタンをクリックしてログファイルをダウンロードします。

[ここでライブ例を表示](https://app.wandb.ai/stacey/deep-drive/runs/pr0os44x/logs)。

![](/images/app_ui/wandb_run_page_log_tab.png)

## Filesタブ
**Filesタブ**を使用して特定のrunに関連付けられたファイルを表示します。モデルチェックポイント、検証セットの例などを保持します。

[ここでライブ例を表示](https://app.wandb.ai/stacey/deep-drive/runs/pr0os44x/files/media/images)。

![](/images/app_ui/wandb_run_page_files_tab.png)

:::tip
W&Bの[Artifacts](../../artifacts/intro.md)を使用してrunの入力と出力を追跡します。Artifactsクイックスタートは[こちら](../../artifacts/artifacts-walkthrough.md)をチェックしてください。
:::

## Artifactsタブ
Artifactsタブには、指定されたrunに対する入力および出力[Artifacts](../../artifacts/intro.md)がリストされます。

[ここでライブ例を表示](https://wandb.ai/stacey/artifact_july_demo/runs/2cslp2rt/artifacts)。

![](/images/app_ui/artifacts_tab.png)