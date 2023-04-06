export type ConcurrencyLimiter = {
  addTask: (fn: () => Promise<void>) => Promise<void>;
};

export function createConcurrencyLimiter(
  maxConcurrent: number
): ConcurrencyLimiter {
  const promiseMap = new Map<string, Promise<void>>();

  return {addTask};

  async function addTask(fn: () => Promise<void>): Promise<void> {
    const promiseKey = `${Math.random()}`;
    while (promiseMap.size >= maxConcurrent) {
      await Promise.race(promiseMap.values());
    }
    promiseMap.set(
      promiseKey,
      (async () => {
        await fn();
        promiseMap.delete(promiseKey);
      })()
    );
  }
}
