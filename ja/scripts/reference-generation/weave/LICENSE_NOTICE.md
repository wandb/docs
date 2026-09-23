---
title: ライセンスに関する通知
---

# リファレンスドキュメント生成に関するライセンス通知 {#license-notice-for-reference-documentation-generation}

## 概要 {#overview}

このディレクトリ内のスクリプトは、開発/CI プロセス中にリファレンスドキュメントを生成するためにのみ使用されます。Weave ライブラリとともに配布されることはなく、本番コードにも含まれません。

## 依存関係と各ライセンス {#dependencies-and-their-licenses}

### 直接依存関係 {#direct-dependencies}

* **requests** (Apache-2.0): HTTPリクエストに使用されます
* **lazydocs** (MIT): W&amp;Bが管理しているドキュメント生成ツール

### 推移的依存関係 (lazydocs 経由) {#transitive-dependencies-via-lazydocs} 

* **setuptools** (MIT、LGPL-3.0 コンポーネントをベンダリングして含む) : ビルドシステム
* ライセンスが混在するその他のさまざまな依存関係

## 重要な注意事項 {#important-notes}

1. **開発時のみ**: これらの依存関係は、CI/GitHub Actions でドキュメントを生成する際に一時的にのみインストールされます。配布される Weave パッケージに含まれることはありません。

2. **配布なし**: 生成されるドキュメントは、実行可能なコードや依存関係を含まない MDX/Markdown ファイルのみで構成されます。

3. **隔離された実行**: GitHub Action は、使用後に破棄される隔離された仮想環境でこれらのスクリプトを実行します。

4. **ライセンス遵守**: これらのツールは Weave と一緒に配布されないため、setuptools の vendored dependencies に含まれる LGPL-3.0 コンポーネントによって、Weave ユーザーにライセンス上の義務が生じることはありません。

## 厳格なライセンスポリシーを持つ組織向け {#for-organizations-with-strict-license-policies}

組織で、開発ツールに LGPL コードが少しでも含まれることを許可しないポリシーを採用している場合:

1. GitHub Action を使用してクラウド上でドキュメントを生成する (推奨) 
2. lazydocs を回避する最小構成の Python ジェネレーターを使用する
3. Docker コンテナー内でドキュメントを生成する
4. 開発専用ツールに対する例外を申請する

## Socket Security {#socket-security}

リポジトリのルートにある `.socketignore` ファイルでは、これらのスクリプトは本番コードではなく開発用ツールであるため、セキュリティスキャンの対象外としています。

### 既知の Socket Security の警告 {#known-socket-security-warnings}

* **wheel に含まれるネイティブコード**: `wheel` パッケージにはネイティブコードが含まれていますが、これは Python のパッケージングツールでは一般的です
* **ライセンス違反**: 一部の推移的依存関係には、LGPL など、ポリシー警告の対象となるライセンスが含まれている場合があります

これらの警告が許容できる理由は次のとおりです。

1. これらのツールはドキュメント生成時にのみ使用されます
2. 分離された CI 環境で実行されます
3. Weave と一緒に配布されることはありません
4. 生成されたドキュメントには実行可能なコードが含まれません