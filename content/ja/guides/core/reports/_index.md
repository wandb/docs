---
title: Reports
description: 機械学習プロジェクトのためのプロジェクト管理とコラボレーションツール
cascade:
- url: guides/reports/:filename
menu:
  default:
    identifier: ja-guides-core-reports-_index
    parent: core
url: guides/reports
weight: 3
---

{{< cta-button productLink="https://wandb.ai/stacey/deep-drive/reports/The-View-from-the-Driver-s-Seat--Vmlldzo1MTg5NQ?utm_source=fully_connected&utm_medium=blog&utm_campaign=view+from+the+drivers+seat" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Report_API_Quickstart.ipynb" >}}

W&B Reports を使用して以下を行います:
- Runs を整理。
- 可視化を埋め込み、自動化。
- 学びを説明。
- Collaborators と LaTeX zip ファイルまたは PDF 形式でアップデートを共有。

この画像は、トレーニング中に W&B にログされたメトリクスから作成されたレポートの一部を示しています。

{{< img src="/images/reports/safe-lite-benchmark-with-comments.png" alt="" max-width="90%" >}}

上記の画像が含まれているレポートを [こちら](https://wandb.ai/stacey/saferlife/reports/SafeLife-Benchmark-Experiments--Vmlldzo0NjE4MzM) で確認できます。

## 仕組み

数クリックで協力的なレポートを作成。

1. W&B アプリ内の W&B プロジェクトワークスペースに移動します。
2. ワークスペースの右上隅にある **Create report** ボタンをクリックします。

{{< img src="/images/reports/create_a_report_button.png" alt="" max-width="90%">}}

3. **Create Report** というモーダルが表示されます。レポートに追加したいチャートやパネルを選択します。（チャートやパネルは後で追加や削除が可能です）。
4. **Create report** をクリックします。
5. レポートを希望の状態に編集します。
6. **Publish to project** をクリックします。
7. **Share** ボタンを使って、Collaborators とレポートを共有します。

W&B Python SDK を使用してインタラクティブにまたはプログラムでレポートを作成する詳細については、[Create a report]({{< relref path="./create-a-report.md" lang="ja" >}}) ページを参照してください。

## 開始方法

ユースケースに応じて、W&B Reports を始めるための以下のリソースを探索してください:

* W&B Reports の概要をつかむために[ビデオデモンストレーション](https://www.youtube.com/watch?v=2xeJIv_K_eI)をチェックしてください。
* ライブ報告の例を見るために、[Reports ギャラリー]({{< relref path="./reports-gallery.md" lang="ja" >}}) を探索してください。
* ワークスペースの作成とカスタマイズ方法を学ぶために [Programmatic Workspaces]({{< relref path="/tutorials/workspaces.md" lang="ja" >}}) チュートリアルを試してください。
* [W&B Fully Connected](http://wandb.me/fc) で curated Reports を読んでください。