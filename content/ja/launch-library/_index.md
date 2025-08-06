---
title: ローンンチ-ライブラリ
menu:
  launch:
    identifier: launch-library
type: docs
hidden: true
cascade:
  type: docs
  hidden: true
  menu:
    launch-library:
      parent: launch-library
---

## クラス

[`class LaunchAgent`](./launchagent.md): 指定された run キューをポーリングして、wandb launch 用の run を起動する Launch エージェントクラスです。

## 関数

[`launch(...)`](./launch.md): W&B launch 実験を実行します。

[`launch_add(...)`](./launch_add.md): W&B launch 実験をキューに追加します。source uri、job、または docker_image のいずれかで指定できます。