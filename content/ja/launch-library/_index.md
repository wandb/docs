---
title: ローンンチ-ライブラリ
cascade:
  hidden: true
  menu:
    launch-library:
      parent: launch-library
  type: docs
hidden: true
menu:
  launch:
    identifier: launch-library
type: docs
---

## クラス

[`class LaunchAgent`](./launchagent.md)：Launch エージェントクラス。指定された run キューをポーリングし、W&B Launch 用の run を起動します。

## 関数

[`launch(...)`](./launch.md)：W&B Launch 実験を開始します。

[`launch_add(...)`](./launch_add.md)：W&B Launch 実験をキューに追加します。source uri、job、または docker_image のいずれかを指定できます。