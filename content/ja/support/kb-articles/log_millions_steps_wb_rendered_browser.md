---
title: 何百万ものステップを W&B にログするとどうなりますか？ブラウザ上ではどのようにレンダリングされますか？
menu:
  support:
    identifier: ja-support-kb-articles-log_millions_steps_wb_rendered_browser
support:
  - experiments
toc_hide: true
type: docs
url: /ja/support/:filename
---
グラフに送信されるポイントの数は、UI の読み込み時間に影響を与えます。1,000 ポイントを超えるラインでは、バックエンドがデータを 1,000 ポイントにサンプリングしてからブラウザに送信します。このサンプリングは非決定論的であり、ページを更新するたびに異なるサンプルポイントとなります。

1 メトリクスあたり 10,000 ポイント未満のログを記録してください。1 行に 100 万ポイントを超えてログを記録すると、ページの読み込み時間が大幅に増加します。この [Colab](http://wandb.me/log-hf-colab) で、正確性を犠牲にせずにログのフットプリントを最小化するための戦略を探ってください。500 列以上の config とサマリーメトリクスがある場合、テーブルに表示されるのは 500 列のみです。