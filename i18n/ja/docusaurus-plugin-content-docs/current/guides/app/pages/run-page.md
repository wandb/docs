---
description: 
  各モデルのトレーニングrunは、専用のページが割り当てられ、大規模なプロジェクト内で整理されます。
---

# Runページ

Runページを使用して、モデルの1つのバージョンに関する詳細情報を調べます。

## 概要タブ

* Run名、説明、タグ
* Runの状態
  * **終了**: スクリプトが終了し、データが完全に同期されたか、`wandb.finish()`が呼び出された
  * **失敗**: スクリプトがゼロ以外の終了ステータスで終了した
  * **クラッシュ**: スクリプトが内部プロセスでハートビートの送信を停止した。これは、マシンがクラッシュする場合に起こることがある
  * **実行中**: スクリプトが実行中で、最近ハートビートを送信している
* ホスト名、オペレーティングシステム、Pythonバージョン、およびRunを開始したコマンド
* [`wandb.config`](../../../guides/track/config.md)で保存された設定パラメータのリスト
* [`wandb.log()`](../../../guides/track/log/intro.md)で保存されたサマリパラメータのリスト（デフォルトでは最後にログに記録された値に設定）

[ライブ例を見る →](https://app.wandb.ai/carey/pytorch-cnn-fashion/runs/munu5vvg/overview?workspace=user-carey)

![W&B Dashboard run overview tab](/images/app_ui/wandb_run_overview_page.png)

Pythonの詳細はプライベートであり、ページ自体を公開しても非公開です。これは、無記名モードの左ページと、私のアカウントの右ページの例です。

![](/images/app_ui/wandb_run_overview_page_2.png)
## チャートタブ

* 可視化の検索、グループ化、並べ替えが可能
  * 検索バーでは正規表現が使えます
* グラフ上の鉛筆アイコン✏️をクリックして編集
  * x軸、メトリクス、範囲の変更が可能
  * 凡例、タイトル、チャートの色を編集
* 検証セットからの例の予測を表示
* これらのチャートを取得するには、[`wandb.log()`](../../../guides/track/log/intro.md)でデータをログに記録して下さい

![](/images/app_ui/wandb-run-page-workspace-tab.png)

## システムタブ

* CPU使用率、システムメモリ、ディスクI/O、ネットワークトラフィック、GPU使用率、GPU温度、GPUメモリアクセス時間、GPUメモリ割り当て、GPU電力消費を可視化
* Lambda LabsはW&Bシステムメトリクスの使用方法を[ブログ記事→](https://lambdalabs.com/blog/weights-and-bias-gpu-cpu-utilization/)で紹介しています。

[ライブ例を見る→](https://wandb.ai/stacey/deep-drive/runs/ki2biuqy/system?workspace=user-carey)

![](/images/app_ui/wandb_system_utilization.png)

## モデルタブ

* モデルのレイヤー、パラメータの数、各レイヤーの出力形状を確認できます

[ライブ例を見る→](https://app.wandb.ai/stacey/deep-drive/runs/pr0os44x/model)

![](/images/app_ui/wandb_run_page_model_tab.png)

## ログタブ
* モデルをトレーニングするマシンからのコマンドライン、stdout、stderrに出力された情報

* 最後の1000行を表示しています。runが終了した後、完全なログファイルをダウンロードしたい場合は、右上のダウンロードボタンをクリックしてください。

[ライブ例を見る →](https://app.wandb.ai/stacey/deep-drive/runs/pr0os44x/logs)

![](/images/app_ui/wandb_run_page_log_tab.png)

## ファイルタブ

* [`wandb.save()`](../../track/save-restore.md)を使用して、runと同期するファイルを保存します。

* モデルのチェックポイント、検証セットの例などを保持します。

* `diff.patch`を使用して、コードの正確なバージョンを[復元](../../track/save-restore.md)します。

  [ライブ例を見る →](https://app.wandb.ai/stacey/deep-drive/runs/pr0os44x/files/media/images)

:::info

W&B [アーティファクト](../../artifacts/intro.md) システムは、データセットやモデルなどの大きなファイルを扱ったり、バージョン管理を行ったり、重複を削除するための追加機能を提供しています。 runの入力および出力をトラッキングするためには、`wandb.save`ではなくアーティファクトを使用することをお勧めします。 アーティファクトのクイックスタートは[こちら](../../artifacts/quickstart.md)からご覧ください。

:::

![](/images/app_ui/wandb_run_page_files_tab.png)

## アーティファクトタブ

* このrunに関連する入力および出力の[アーティファクト](../../artifacts/intro.md)の検索可能なリストを提供します。

* 行をクリックすると、このrunで使用された、または生成された特定のアーティファクトに関する情報が表示されます。

* ウェブアプリのアーティファクトビューアを操作および使用する方法については、[プロジェクト](project-page.md)レベルの[アーティファクトタブ](project-page.md#artifacts-tab)の参照をご覧ください。[ライブ例を見る →](https://wandb.ai/stacey/artifact_july_demo/runs/2cslp2rt/artifacts)

![](/images/app_ui/artifacts_tab.png)