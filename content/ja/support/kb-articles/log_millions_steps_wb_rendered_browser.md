---
title: What happens when I log millions of steps to W&B? How is that rendered in the
  browser?
menu:
  support:
    identifier: ja-support-kb-articles-log_millions_steps_wb_rendered_browser
support:
- experiments
toc_hide: true
type: docs
url: /support/:filename
---

送信される点の数は、UI でのグラフのロード時間に影響します。1,000 点を超える線については、バックエンドはデータを 1,000 点にサンプリングしてからブラウザに送信します。このサンプリングは非決定的であるため、ページをリフレッシュすると、サンプリングされる点が異なります。

メトリクス ごとに 10,000 ポイント未満を ログ してください。100 万ポイントを超える ログ を 1 行で記録すると、ページのロード時間が大幅に長くなります。この [Colab](http://wandb.me/log-hf-colab) で、精度を犠牲にすることなく ログ のフットプリントを最小限に抑えるための戦略を検討してください。config とサマリー メトリクス の列が 500 を超える場合、テーブルに表示されるのは 500 のみです。
