---
title: ジョブやオートメーションのためのシークレットを指定することはできますか？例えば、ユーザーに直接見せたくないAPIキーのようなものですか？
menu:
  launch:
    identifier: ja-launch-launch-faq-secrets_jobsautomations_instance_api_key_wish_directly_visible
    parent: launch-faq
---

はい。次の手順に従ってください：

1. run 用の指定された名前空間に Kubernetes のシークレットを作成します。コマンドは以下の通りです：
   `kubectl create secret -n <namespace> generic <secret_name> <secret_value>`

2. シークレットを作成したら、run が開始する際にシークレットを注入するようにキューを設定します。クラスター管理者だけがシークレットを見ることができ、エンドユーザーはそれを確認できません。