---
title: Reports
description: 機械学習 プロジェクト のための プロジェクト 管理とコラボレーション ツール
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

W&B Reports の用途:
- Runs の整理。
- 可視化の埋め込みと自動化。
- 学び の記述。
- コラボレーターとの更新情報の共有 (LaTeX zip ファイルまたは PDF として)。

以下の画像は、トレーニングの過程で W&B に ログ された メトリクス から作成された report のセクションを示しています。

{{< img src="/images/reports/safe-lite-benchmark-with-comments.png" alt="" max-width="90%" >}}

上記の画像が取得された report は [こちら](https://wandb.ai/stacey/saferlife/reports/SafeLife-Benchmark-Experiments--Vmlldzo0NjE4MzM) からご覧ください。

## 仕組み
数回クリックするだけで、コラボレーティブ report を作成できます。

1. W&B App で W&B project の workspace に移動します。
2. workspace の右上隅にある [**report を作成**] ボタンをクリックします。

{{< img src="/images/reports/create_a_report_button.png" alt="" max-width="90%">}}

3. [**Report を作成**] というタイトルのモーダルが表示されます。 report に追加するグラフと パネル を選択します (グラフと パネル は後で追加または削除できます)。
4. [**report を作成**] をクリックします。
5. report を希望の状態に編集します。
6. [**project に公開**] をクリックします。
7. [**共有**] ボタンをクリックして、コラボレーターと report を共有します。

W&B Python SDK を使用して report をインタラクティブに、またプログラムで作成する方法の詳細については、[report の作成]({{< relref path="./create-a-report.md" lang="ja" >}}) ページを参照してください。

## 開始方法
ユースケース に応じて、以下のリソースを調べて W&B Reports を開始してください。

* W&B Reports の概要については、[ビデオ デモ](https://www.youtube.com/watch?v=2xeJIv_K_eI) をご覧ください。
* ライブ report の例については、[Reports ギャラリー]({{< relref path="./reports-gallery.md" lang="ja" >}}) を調べてください。
* workspace を作成およびカスタマイズする方法については、[プログラムによる Workspaces]({{< relref path="/tutorials/workspaces.md" lang="ja" >}}) チュートリアルをお試しください。
* [W&B Fully Connected](http://wandb.me/fc) で厳選された Reports をお読みください。
