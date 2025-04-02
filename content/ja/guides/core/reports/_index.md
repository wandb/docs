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
- コラボレーターとの更新情報の共有（LaTeX zip ファイルまたは PDF として）。




次の画像は、トレーニングの過程で W&B に ログ された メトリクス から作成された レポート のセクションを示しています。

{{< img src="/images/reports/safe-lite-benchmark-with-comments.png" alt="" max-width="90%" >}}

上記の画像が取得された レポート は [こちら](https://wandb.ai/stacey/saferlife/reports/SafeLife-Benchmark-Experiments--Vmlldzo0NjE4MzM) からご覧ください。

## 仕組み
数回クリックするだけで、コラボレーティブな レポート を作成できます。

1. W&B App で W&B project workspace に移動します。
2. workspace の右上隅にある [**Create report**] (レポート の作成) ボタンをクリックします。

{{< img src="/images/reports/create_a_report_button.png" alt="" max-width="90%">}}

3. [**Create Report**] (レポート の作成) という タイトル のモーダルが表示されます。 レポート に追加するグラフと パネル を選択します (グラフと パネル は後で追加または削除できます)。
4. [**Create report**] (レポート の作成) をクリックします。
5. レポート を希望の状態に編集します。
6. [**Publish to project**] (project に公開) をクリックします。
7. [**Share**] (共有) ボタンをクリックして、コラボレーターと レポート を共有します。

W&B Python SDK を使用して インタラクティブ およびプログラムで レポート を作成する方法の詳細については、[レポート の作成]({{< relref path="./create-a-report.md" lang="ja" >}}) ページを参照してください。

## 開始方法
ユースケース に応じて、次のリソースを調べて W&B Reports を開始してください。

* W&B Reports の概要については、[ビデオ デモ](https://www.youtube.com/watch?v=2xeJIv_K_eI) をご覧ください。
* ライブ レポート の例については、[Reports gallery]({{< relref path="./reports-gallery.md" lang="ja" >}}) をご覧ください。
* [Programmatic Workspaces]({{< relref path="/tutorials/workspaces.md" lang="ja" >}}) チュートリアルを試して、workspace の作成方法とカスタマイズ方法を学んでください。
* [W&B Fully Connected](http://wandb.me/fc) で厳選された Reports を読んでください。

## 推奨される ベストプラクティス とヒント

Experiments と ログ の ベストプラクティス とヒントについては、[Best Practices: Reports](https://wandb.ai/wandb/pytorch-lightning-e2e/reports/W-B-Best-Practices-Guide--VmlldzozNTU1ODY1#reports) を参照してください。
