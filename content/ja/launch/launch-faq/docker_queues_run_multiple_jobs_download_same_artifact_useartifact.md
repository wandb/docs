---
title: 複数の Docker キュー内のジョブが同じ Artifact をダウンロードする場合、キャッシュは使われますか？それとも毎回再ダウンロードされますか？
menu:
  launch:
    identifier: ja-launch-launch-faq-docker_queues_run_multiple_jobs_download_same_artifact_useartifact
    parent: launch-faq
---

キャッシュ機能はありません。各 Launch ジョブは独立して動作します。キューやエージェントで共有キャッシュを利用する場合は、キューの設定で Docker の引数を使ってマウントしてください。

さらに、特定のユースケースでは、W&B Artifacts のキャッシュを永続的なボリュームとしてマウントすることもできます。