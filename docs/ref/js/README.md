---
description: TypeScript, Node 및 최신 웹 브라우저용 W&B SDK
---

# JavaScript Library

Python 라이브러리와 비슷하게, JavaScript/TypeScript에서 실험을 추적할 수 있는 클라이언트를 제공합니다.

- Node 서버에서 메트릭을 로그하고 이를 W&B에서 인터랙티브한 플롯으로 표시
- 인터랙티브 트레이스를 통해 LLM 애플리케이션 디버깅
- [LangChain.js](https://github.com/hwchase17/langchainjs) 사용 디버깅

이 라이브러리는 Node 및 최신 JS 런타임과 호환됩니다.

JavaScript 클라이언트의 소스 코드는 [Github repository](https://github.com/wandb/wandb-js)에서 찾을 수 있습니다.

:::info
JavaScript 인테그레이션은 아직 베타 버전이므로, 문제가 발생하면 알려주세요!
:::

### Installation

```shell
npm install @wandb/sdk
# or ...
yarn add @wandb/sdk
```

### Usage

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
우리는 비동기로 모든 API 호출을 처리하기 위해 별도의 MessageChannel을 생성합니다. `await wandb.finish()`를 호출하지 않으면 스크립트가 중단됩니다.
:::

Node/CommonJS:

```javascript
const wandb = require('@wandb/sdk').default;
```

현재, Python SDK에서 제공하는 많은 기능이 아직 없지만 기본적인 로그 기능은 사용 가능합니다. [Tables](/guides/tables?utm_source=github&utm_medium=code&utm_campaign=wandb&utm_content=readme)와 같은 추가 기능도 곧 추가될 예정입니다.

### Authentication and Settings

노드 환경에서는 `process.env.WANDB_API_KEY`를 찾고 TTY가 있으면 입력을 요청합니다. 비노드 환경에서는 `sessionStorage.getItem("WANDB_API_KEY")`를 찾습니다. 추가 설정은 [여기에서](https://github.com/wandb/wandb-js/blob/main/src/sdk/lib/config.ts) 확인할 수 있습니다.

# Integrations

우리의 [Python 인테그레이션](/guides/integrations)은 커뮤니티에서 널리 사용되고 있으며, LLM 앱 빌더들이 원하는 툴을 활용할 수 있도록 더 많은 JavaScript 인테그레이션을 구축하려고 합니다.

추가 인테그레이션에 대한 요청이 있다면, 요청의 자세한 내용을 포함하여 이슈를 열어 주시면 감사하겠습니다.

## LangChain.js

이 라이브러리는 LLM 애플리케이션을 구축하기 위한 인기 있는 라이브러리 [LangChain.js](https://github.com/hwchase17/langchainjs) 버전 >= 0.0.75와 통합됩니다.

### Usage

```typescript
import {WandbTracer} from '@wandb/sdk/integrations/langchain';

const wbTracer = await WandbTracer.init({project: 'langchain-test'});
// langchain 워크로드를 실행하세요...
chain.call({input: "My prompt"}, wbTracer)
await WandbTracer.finish();
```

:::caution
우리는 비동기로 모든 API 호출을 처리하기 위해 별도의 MessageChannel을 생성합니다. `await WandbTracer.finish()`를 호출하지 않으면 스크립트가 중단됩니다.
:::

보다 자세한 예시는 [이 테스트](https://github.com/wandb/wandb-js/blob/main/src/sdk/integrations/langchain/langchain.test.ts)에서 확인하세요.