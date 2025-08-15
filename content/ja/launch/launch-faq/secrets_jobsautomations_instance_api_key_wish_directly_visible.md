---
title: ジョブやオートメーションでシークレットを指定できますか？例えば、ユーザーに直接見せたくない APIキー などは指定できますか？
menu:
  launch:
    identifier: ja-launch-launch-faq-secrets_jobsautomations_instance_api_key_wish_directly_visible
    parent: launch-faq
---

はい。以下の手順に従ってください。

1. run 用の指定された namespace に Kubernetes シークレットを作成します。コマンドは以下の通りです:  
   `kubectl create secret -n <namespace> generic <secret_name> <secret_value>`

2. シークレットを作成した後、queue が run 開始時にそのシークレットをインジェクトするように設定します。シークレットはクラスター管理者のみが閲覧可能で、エンドユーザーからは見えません。