---
description: Javaクライアント・ライブラリの概要
---

# Javaライブラリ

Pythonライブラリと同様に、Javaクライアントを利用して機械学習モデルを計測し、実験をトラッキングすることができます。このライブラリは、Pythonライブラリをラップするために使用される2つのシンプルなクラスで構成されています。

Javaクライアントのソースコードは、[Githubリポジトリ](https://github.com/wandb/client-ng-java)で見つけることができます。

:::info
Javaの統合はまだベータ版です。問題が発生した場合はお知らせください！
:::

### インストール

1. wandb Pythonクライアントの最新バージョンをインストールします: `pip install wandb[grpc] --upgrade`
2. JavaプロジェクトにWandbのjarファイルを含めるだけです。

    **Maven**: `pom.xml`ファイルにjarファイルを追加することで含めることができます。
    メーベンリポジトリを使用する場合：

    ```markup
    <dependency>
        <groupId>com.wandb.client</groupId>
        <artifactId>client-ng-java</artifactId>
        <version>1.0-SNAPSHOT</version>
    </dependency>
    ```

    または、[Githubのパッケージ](https://github.com/wandb/client-ng-java/packages/381057)からjarファイルを直接ダウンロードすることができます：

    ```markup
    <dependencies>
        <dependency>
            <groupId>com.wandb.client</groupId>
            <artifactId>client-ng-java</artifactId>
            <version>1.0-SNAPSHOT</version>
            <scope>system</scope>
            <systemPath>/root/path/to/jar/file.jar</systemPath>
        </dependency>
    </dependencies>
    ```