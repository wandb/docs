---
title: W&B に何百万ものステップをログした場合、ブラウザではどのように表示されますか？
menu:
  support:
    identifier: ja-support-kb-articles-log_millions_steps_wb_rendered_browser
support:
- 実験
toc_hide: true
type: docs
url: /support/:filename
---

送信されるポイント数が多いほど、UI のグラフの読み込み時間に影響します。1,000 ポイントを超える線については、バックエンドがデータを 1,000 ポイントまでサンプリングしてブラウザに送信します。このサンプリングは非決定的であり、ページをリフレッシュするたびに異なるサンプルポイントが表示されます。

1 メトリクスにつき 10,000 ポイント未満のログが推奨です。1 ラインで 100 万ポイント以上をログすると、ページの読み込み時間が大幅に増加します。正確性を犠牲にせず、ログの負荷を最小限に抑える戦略をこの [Colab](https://wandb.me/log-hf-colab) でご確認ください。config や summary metrics のカラム数が 500 を超えた場合、テーブルには最大 500 カラムのみが表示されます。