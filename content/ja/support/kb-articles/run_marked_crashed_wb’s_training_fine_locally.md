---
title: ローカルでは問題なくトレーニングされているのに、なぜ W&B で run が「crashed」と表示されるのですか？
menu:
  support:
    identifier: ja-support-kb-articles-run_marked_crashed_wb’s_training_fine_locally
support:
- クラッシュやハングしている run
toc_hide: true
type: docs
url: /support/:filename
---

これは接続の問題を示しています。サーバーがインターネットへのアクセスを失い、データが W&B への同期を停止すると、システムは短い再試行期間の後に run をクラッシュしたとマークします。