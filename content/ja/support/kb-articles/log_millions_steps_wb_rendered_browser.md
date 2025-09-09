---
title: W&B に 数百万ステップを ログ した場合はどうなりますか？ ブラウザではどのように表示されますか？
menu:
  support:
    identifier: ja-support-kb-articles-log_millions_steps_wb_rendered_browser
support:
- 実験
toc_hide: true
type: docs
url: /support/:filename
---

送信するポイント数は UI のグラフの読み込み時間に影響します。1,000 ポイントを超える線については、バックエンドがブラウザに送る前にデータを 1,000 ポイントにまでサンプリングします。このサンプリングは非決定的で、ページを更新するたびにサンプルされるポイントが異なる場合があります。

各メトリクスのログは 10,000 ポイント未満に抑えてください。1 本の線で 100 万ポイントをログすると、ページの読み込み時間が大幅に増加します。精度を損なわずにログのフットプリントを最小化する戦略は、この [Colab](https://wandb.me/log-hf-colab) で確認できます。config と summary メトリクスの列が 500 を超える場合、テーブルに表示されるのは 500 列のみです。