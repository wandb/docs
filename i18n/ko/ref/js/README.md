---
description: The W&B SDK for TypeScript, Node, and modern Web Browsers
---

# JavaScript 라이브러리

우리의 Python 라이브러리와 유사하게, JavaScript/TypeScript에서 실험을 추적하기 위한 클라이언트를 제공합니다.

- Node 서버에서 메트릭을 로그하고 W&B에서 인터랙티브 플롯으로 표시합니다
- 인터랙티브 트레이스로 LLM 애플리케이션 디버깅
- [LangChain.js](https://github.com/hwchase17/langchainjs) 사용 디버깅

이 라이브러리는 Node와 현대 JS 런타임과 호환됩니다.

JavaScript 클라이언트의 소스 코드는 [Github 저장소](https://github.com/wandb/wandb-js)에서 찾을 수 있습니다.

:::info
우리의 JavaScript 인테그레이션은 아직 베타 버전이므로, 문제가 발생하면 알려주세요!
:::

### 설치

```shell
npm install @wandb/sdk
# 또는 ...
yarn add @wandb/sdk
```

### 사용법

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
모든 API 호출을 비동기적으로 처리하기 위해 별도의 MessageChannel을 생성합니다. `await wandb.finish()`를 호출하지 않으면 스크립트가 멈추게 됩니다.
:::

Node/CommonJS:

```javascript
const wandb = require('@wandb/sdk').default;
```

우리의 Python SDK에서 찾을 수 있는 많은 기능이 현재 부족하지만, 기본 로깅 기능을 사용할 수 있습니다. 곧 [Tables](https://docs.wandb.ai/guides/tables?utm_source=github&utm_medium=code&utm_campaign=wandb&utm_content=readme)와 같은 추가 기능을 추가할 예정입니다.

### 인증 및 설정

node 환경에서는 `process.env.WANDB_API_KEY`를 찾아보고 TTY가 있는 경우 입력을 요청합니다. node가 아닌 환경에서는 `sessionStorage.getItem("WANDB_API_KEY")`를 찾습니다. 추가 설정은 [여기에서 확인](https://github.com/wandb/wandb-js/blob/main/src/sdk/lib/config.ts)할 수 있습니다.

# 인테그레이션

우리의 [Python 인테그레이션](https://docs.wandb.ai/guides/integrations)은 커뮤니티에서 널리 사용되며, LLM 앱 빌더들이 원하는 툴을 활용할 수 있도록 더 많은 JavaScript 인테그레이션을 구축하기를 희망합니다.

추가 인테그레이션에 대한 요청이 있으시다면, 요청에 대한 세부 정보와 함께 이슈를 열어 주시기 바랍니다.

## LangChain.js

이 라이브러리는 LLM 애플리케이션 구축을 위한 인기 라이브러리인 [LangChain.js](https://github.com/hwchase17/langchainjs) 버전 >= 0.0.75과 통합됩니다.

### 사용법

```typescript
import {WandbTracer} from '@wandb/sdk/integrations/langchain';

const wbTracer = await WandbTracer.init({project: 'langchain-test'});
// 랭체인 작업을 실행하세요...
chain.call({input: "My prompt"}, wbTracer)
await WandbTracer.finish();
```

:::caution
모든 API 호출을 비동기적으로 처리하기 위해 별도의 MessageChannel을 생성합니다. `await WandbTracer.finish()`를 호출하지 않으면 스크립트가 멈추게 됩니다.
:::

보다 자세한 예제는 [이 테스트](https://github.com/wandb/wandb-js/blob/main/src/sdk/integrations/langchain/langchain.test.ts)를 참고하세요.