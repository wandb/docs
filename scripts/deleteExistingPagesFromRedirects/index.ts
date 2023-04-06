import axios from 'axios';
import yargs from 'yargs';

import {loadData} from './data';
import {createConcurrencyLimiter} from './concurrency';

const BASE_URL = 'https://docs.wandb.ai';

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
const concurrencyLimiter = createConcurrencyLimiter(5);

(async () => {
  let count = 0;
  for (const {from, to} of data.redirects) {
    concurrencyLimiter.addTask(async () => {
      const fromPageExists = await pageExists(from);
      if (fromPageExists) {
        count++;
        console.log(from);
      }
    });
  }

  console.log(
    `\n${count} / ${data.redirects.length} redirects from pages that already exist`
  );
})();

async function pageExists(path: string): Promise<boolean> {
  const {status} = await axios.get(`${BASE_URL}${path}`, {
    validateStatus: null,
  });
  return status === 200;
}
