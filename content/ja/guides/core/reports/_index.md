---
title: Reports
description: 機械学習プロジェクト向けのプロジェクト管理・コラボレーションツール
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

W&B Reports を使用して、以下を行うことができます。
- Runs を整理する。
- 可視化を埋め込み、自動化する。
- 学びを記述する。
- コラボレーターと、LaTeX zip ファイルまたは PDF として更新を共有する。

以下の画像は、トレーニング中に W&B にログされたメトリクスから作成されたレポートの一部を示しています。

{{< img src="/images/reports/safe-lite-benchmark-with-comments.png" alt="ベンチマーク結果を含む W&B Reports" max-width="90%" >}}

上記の画像が引用されたレポートを[こちら](https://wandb.ai/stacey/saferlife/reports/SafeLife-Benchmark-Experiments--Vmlldzo0NjE4MzM)でご覧ください。

## 仕組み
数回のクリックで共同レポートを作成できます。

1. W&B App で W&B Project Workspace に移動します。
2. Workspace の右上にある **Create report** ボタンをクリックします。

{{< img src="/images/reports/create_a_report_button.png" alt="レポート作成ボタン" max-width="90%">}}

3. **Create Report** と題されたモーダルが表示されます。レポートに追加したいチャートとパネルを選択します。（チャートとパネルは後で追加または削除できます。）
4. **Create report** をクリックします。
5. レポートを希望の状態に編集します。
6. **Publish to project** をクリックします。
7. **Share** ボタンをクリックして、レポートをコラボレーターと共有します。

W&B Python SDK を使用して、対話的およびプログラム的にレポートを作成する方法の詳細については、[レポートの作成]({{< relref path="./create-a-report.md" lang="ja" >}})ページを参照してください。

## 開始方法
ユースケースに応じて、W&B Reports の開始に役立つ以下のリソースを参照してください。

* W&B Reports の概要を把握するには、[ビデオデモンストレーション](https://www.youtube.com/watch?v=2xeJIv_K_eI)をご覧ください。
* ライブ Reports の例については、[Reports ギャラリー]({{< relref path="./reports-gallery.md" lang="ja" >}})を参照してください。
* Workspace の作成方法とカスタマイズ方法を学ぶには、[Programmatic Workspaces]({{< relref path="/tutorials/workspaces.md" lang="ja" >}}) チュートリアルをお試しください。
* [W&B Fully Connected](https://wandb.me/fc)で厳選された Reports をお読みください。

## 推奨プラクティスとヒント

Experiments およびロギングのベストプラクティスとヒントについては、[ベストプラクティス: Reports](https://wandb.ai/wandb/pytorch-lightning-e2e/reports/W-B-Best-Practices-Guide--VmlldzozNTU1ODY1#reports) を参照してください。