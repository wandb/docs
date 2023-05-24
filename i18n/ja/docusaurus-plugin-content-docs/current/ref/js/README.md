---
description: TypeScriptやNode、現代のウェブブラウザ用のW&B SDK
---

# JavaScriptライブラリ

Pythonライブラリと同様に、JavaScript/TypeScriptで実験をトラッキングするためのクライアントを提供しています。

- Nodeサーバーからメトリクスをログし、W&Bでインタラクティブな精度図を表示
- 対話式トレースを使用してLLMアプリケーションのデバッグ
- [LangChain.js](https://github.com/hwchase17/langchainjs) の使用デバッグ

このライブラリは、Nodeおよび現代的なJSランタイムと互換性があります。

JavaScriptクライアントのソースコードは、[Githubリポジトリ](https://github.com/wandb/wandb-js)で見つけることができます。

:::info
JavaScript統合はまだBeta版ですので、問題が発生した場合はお知らせください！
:::

### インストール方法

```shell
npm install @wandb/sdk
# または ...
yarn add @wandb/sdk
```

### 使用方法

TypeScript/ESM:

```typescript
import wandb from '@wandb/sdk'

async function track() {
    await wandb.init({config: {test: 1}});
    wandb.log({acc: 0.9, loss: 0.1});
    wandb.log({acc: 0.91, loss: 0.09});
    await wandb.finish();
}

await track()
```

:::caution
すべてのAPI呼び出しを非同期で処理するために別のMessageChannelを生成します。`await wandb.finish()`を呼び出さないと、スクリプトが停止してしまうことがあります。
:::

Node/CommonJS:

```javascript
const wandb = require('@wandb/sdk').default;
```

現在、Python SDKに見られる多くの機能が欠けていますが、基本的なログ機能は利用可能です。近いうちに、[Tables](https://docs.wandb.ai/guides/data-vis?utm_source=github&utm_medium=code&utm_campaign=wandb&utm_content=readme)などの追加機能を追加予定です。

### 認証と設定

Node環境では、`process.env.WANDB_API_KEY`を探し、TTYがある場合は入力を求めます。非Node環境では、`sessionStorage.getItem("WANDB_API_KEY")`を探します。追加の設定は[こちら](https://github.com/wandb/wandb-js/blob/main/src/sdk/lib/config.ts)で見つけることができます。
# インテグレーション

[Pythonインテグレーション](https://docs.wandb.ai/guides/integrations)は、私たちのコミュニティに広く利用されており、より多くのJavaScriptインテグレーションを構築して、LLM アプリビルダーが好きなツールを活用できるようにしたいと考えています。

追加のインテグレーションについてのリクエストがあれば、詳細を記載した issue をオープンしていただけると嬉しいです。

## LangChain.js

このライブラリは、LLMアプリケーションを構築するための人気のあるライブラリである[LangChain.js](https://github.com/hwchase17/langchainjs)と統合されています。

### 利用方法

```typescript
import {WandbTracer} from '@wandb/sdk/integrations/langchain';

await WandbTracer.init({project: 'langchain-test'});

// run your langchain workloads...

await WandbTracer.finish();
```

:::caution
私たちは、すべての API コールを非同期で処理するために、別の MessageChannel をスポーンします。 これにより、`await WandbTracer.finish()` を呼び出さないと、スクリプトがハングアップする可能性があります。
:::

:::caution
トレーサーは並行実行とうまく連携できません。 langchain js の[トレーシングドキュメント](https://js.langchain.com/docs/production/tracing)の最下部にある例を参照してください。
:::

より詳細な例については、[このテスト](https://github.com/wandb/wandb-js/blob/main/src/sdk/integrations/langchain/langchain.test.ts)を参照してください。