---
description: 機械学習プロジェクトのためのプロジェクト管理とコラボレーションツール
slug: /guides/reports
displayed_sidebar: default
---

import Translate, {translate} from '@docusaurus/Translate';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';


# Collaborative Reports

<CTAButtons productLink="https://wandb.ai/stacey/deep-drive/reports/The-View-from-the-Driver-s-Seat--Vmlldzo1MTg5NQ?utm_source=fully_connected&utm_medium=blog&utm_campaign=view+from+the+drivers+seat" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Report_API_Quickstart.ipynb"/>

W&B Reports を使って、Runs を整理し、可視化を埋め込み、自動化し、学びを記述し、コラボレーターと共有しましょう。レポートを LaTeX zip ファイルとして簡単にエクスポートしたり、PDF に変換することもできます。

次の画像は、トレーニングの過程で W&B にログされたメトリクスから作成されたレポートの一部を示しています。

![](/images/reports/safe-lite-benchmark-with-comments.png)

上記の画像が含まれたレポートは [こちら](https://wandb.ai/stacey/saferlife/reports/SafeLife-Benchmark-Experiments--Vmlldzo0NjE4MzM)から確認できます。

## 仕組み
数回のクリックでコラボレーティブレポートを作成します。

1. W&B アプリで W&B プロジェクトワークスペースに移動します。
2. ワークスペースの右上にある **Create report** ボタンをクリックします。

![](/images/reports/create_a_report_button.png)

3. **Create Report** タイトルのモーダルが表示されます。レポートに追加したいチャートやパネルを選択します。（後でチャートやパネルを追加・削除できます）。
4. **Create report** をクリックします。
5. 希望の状態になるようにレポートを編集します。
6. **Publish to project** をクリックします。
7. **Share** ボタンをクリックして、レポートをコラボレーターと共有します。

W&B Python SDK を使用して、インタラクティブおよびプログラムでレポートを作成する方法については、[Create a report](./create-a-report.md) ページを参照してください。

## 開始方法
ユースケースに応じて、W&B Reports を開始するための以下のリソースを探索してください：

* W&B Reports の概要については、[ビデオデモ](https://www.youtube.com/watch?v=2xeJIv_K_eI)をご覧ください。
* ライブレポートの例については、[Reports gallery](./reports-gallery.md) をご覧ください。
* [W&B Fully Connected](http://wandb.me/fc)でキュレーションされたレポートをお読みください。