---
title: "ライセンスに関する通知"
---

<div id="license-notice-for-reference-documentation-generation">
  # リファレンスドキュメント生成のためのライセンスに関する通知
</div>

<div id="overview">
  ## 概要
</div>

このディレクトリ内のスクリプトは、開発および CI プロセス中にリファレンスドキュメントを生成する目的のみに使用されます。これらは Weave ライブラリの一部として配布されたり、本番コードに含まれたりすることはありません。

<div id="dependencies-and-their-licenses">
  ## 依存関係とライセンス
</div>

<div id="direct-dependencies">
  ### 直接の依存関係
</div>

- **requests** (Apache-2.0): HTTP リクエストの送信に使用
- **lazydocs** (MIT): W&B が保守しているドキュメント生成ツール

<div id="transitive-dependencies-via-lazydocs">
  ### 推移的依存関係（lazydocs 経由）
</div>

- **setuptools**（MIT ライセンス＋組み込みの LGPL-3.0 コンポーネント）：ビルドシステム
- ライセンス形態が混在するその他の依存関係

<div id="important-notes">
  ## 重要な注意事項
</div>

1. **開発用途のみ**: これらの依存関係は、CI/GitHub Actions でのドキュメント生成時に一時的にのみインストールされます。配布される Weave パッケージには一切含まれません。

2. **配布なし**: 生成されるドキュメントは、実行可能なコードや依存関係を一切含まない MDX/Markdown ファイルのみで構成されます。

3. **隔離された実行環境**: GitHub Action は、使用後に破棄される隔離された仮想環境内でこれらのスクリプトを実行します。

4. **ライセンス準拠**: これらのツールは Weave と共に配布されないため、setuptools の同梱依存関係に含まれる LGPL-3.0 コンポーネントは、Weave ユーザーに追加のライセンス上の義務を生じさせません。

<div id="for-organizations-with-strict-license-policies">
  ## 厳格なライセンスポリシーを持つ組織向け
</div>

組織で開発ツールにおける LGPL コードの使用を一切認めないポリシーがある場合は、次のいずれかの方法を検討してください:

1. GitHub Action を使ってクラウド上でドキュメントを生成する（推奨）
2. `lazydocs` を使用しない最小限の Python ジェネレーターを使用する
3. Docker コンテナ内でドキュメントを生成する
4. 開発専用ツールとしての例外を申請する

<div id="socket-security">
  ## ソケットセキュリティ
</div>

リポジトリのルートにある `.socketignore` ファイルでは、これらのスクリプトが本番コードではなく開発用ツールであるため、セキュリティスキャンの対象外としています。

<div id="known-socket-security-warnings">
  ### 既知のソケットセキュリティ警告
</div>

- **wheel 内のネイティブコード**: `wheel` パッケージにはネイティブコードが含まれており、これは Python のパッケージングツールとしては通常の挙動です
- **ライセンス違反**: いくつかの間接依存関係が LGPL などのライセンスを採用しており、ポリシー警告が発生する場合があります

これらの警告が許容される理由は次のとおりです:

1. これらのツールはドキュメント生成時にのみ使用される
2. CI の隔離された環境内で実行される
3. Weave と共に配布されることはない
4. 生成されたドキュメントには実行可能なコードが含まれない