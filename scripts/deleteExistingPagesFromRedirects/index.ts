import axios from 'axios';
import yargs from 'yargs';

import {loadData, removeRedirects} from './data';
import {createConcurrencyLimiter} from './concurrency';
import type {Redirect} from '../utils';

const BASE_URL = 'https://docs-beta.wandb.ai';

const {dataFile} = yargs(process.argv.slice(2))
  .options({
    dataFile: {
      alias: 'd',
      type: 'string',
      description: 'Path to the data JSON file',
    },
  })
  .parseSync();

let data = loadData(dataFile);
const concurrencyLimiter = createConcurrencyLimiter(3);

(async () => {
  const toRemove: Redirect[] = [];

  for (const r of data.redirects) {
    const {from} = r;
    concurrencyLimiter.addTask(async () => {
      const fromPageExists = await pageExists(from);
      if (fromPageExists) {
        toRemove.push(r);
        console.log(from);
      }
    });
  }

  await concurrencyLimiter.waitForTasksToFinish();

  console.log(
    `\n${toRemove.length} / ${data.redirects.length} redirects from pages that already exist`
  );

  data = removeRedirects(data, toRemove);
})();

async function pageExists(path: string): Promise<boolean> {
  const {status} = await axios.get(`${BASE_URL}${path}`, {
    validateStatus: null,
  });
  return status === 200;
}
