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

送信されるポイントの数は、UI でのグラフの読み込み時間に影響します。1,000 ポイントを超える線の場合、バックエンドはデータを 1,000 ポイントにダウンサンプリングしてから、ブラウザに送信します。このサンプリングは非決定的であるため、ページを更新すると、サンプリングされるポイントが異なります。

メトリクス ごとに 10,000 ポイント未満を ログ してください。100 万ポイントを超える ログ を 1 行で記述すると、ページの読み込み時間が大幅に長くなります。この [Colab](http://wandb.me/log-hf-colab) で、精度を犠牲にすることなく ログ のフットプリントを最小限に抑える方法を検討してください。設定とサマリー メトリクス の列が 500 を超える場合、テーブルには 500 のみが表示されます。
