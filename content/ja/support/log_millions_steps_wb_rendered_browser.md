---
title: What happens when I log millions of steps to W&B? How is that rendered in the
  browser?
menu:
  support:
    identifier: ja-support-log_millions_steps_wb_rendered_browser
tags:
- experiments
toc_hide: true
type: docs
---

グラフの読み込み時間は、送信されるポイント数によって影響を受けます。1,000 ポイントを超える線に対しては、バックエンドがデータを 1,000 ポイントにサンプリングしてからブラウザに送信します。このサンプリングは非決定的であり、ページを更新するたびに異なるサンプル化ポイントが表示されます。

1 つのメトリクスにつき 10,000 ポイント未満にログを減らしてください。1 行に 100 万ポイント以上ログすると、ページの読み込み時間が大幅に増加します。精度を損なうことなくログのフットプリントを最小化する方法について、この [Colab](http://wandb.me/log-hf-colab) で探ってみてください。設定や要約メトリクスが 500 列以上ある場合、テーブルには 500 列のみが表示されます。