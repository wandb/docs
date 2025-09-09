---
title: ローカルでは問題なくトレーニングできているのに、W&B で run がクラッシュとしてマークされるのはなぜですか？
menu:
  support:
    identifier: ja-support-kb-articles-run_marked_crashed_wb’s_training_fine_locally
support:
- クラッシュやハングする runs
toc_hide: true
type: docs
url: /support/:filename
---

これは接続に問題があることを示しています。サーバーがインターネット アクセスを失い、データの W&B への同期が停止すると、短い再試行期間の後、システムは run をクラッシュとしてマークします。