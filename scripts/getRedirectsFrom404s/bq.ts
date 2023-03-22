import {BigQuery as BQ} from '@google-cloud/bigquery';

const bqProd = new BQ({projectId: `wandb-production`});

export async function queryBQ<T>(query: string): Promise<T[]> {
  const [job] = await bqProd.createQueryJob(query);
  const [rows] = await job.getQueryResults();
  return rows;
}
