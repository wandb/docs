---
title: When multiple jobs in a Docker queue download the same artifact, is any caching
  used, or is it re-downloaded every run?
menu:
  launch:
    identifier: ja-launch-launch-faq-docker_queues_run_multiple_jobs_download_same_artifact_useartifact
    parent: launch-faq
---

キャッシュは存在しません。各ローンンチジョブは独立して動作します。キュー設定で Docker の引数を使用して、共有キャッシュをマウントするようにキューまたはエージェントを設定します。

さらに、特定のユースケースに W&B アーティファクトのキャッシュを永続ボリュームとしてマウントします。