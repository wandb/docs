import immer, {enableMapSet} from 'immer';

import {
  DEFAULT_DATA_FILE_PATH,
  Data,
  Redirect,
  parseJSONFile,
  sortRedirects,
  writeJSONFile,
} from '../utils';

enableMapSet();

export function addRedirects(
  data: Data,
  redirects: Redirect[],
  dataFilePath = DEFAULT_DATA_FILE_PATH
): Data {
  const newData = immer(data, draft => {
    draft.redirects.push(...redirects);
  });
  saveData(newData, dataFilePath);
  return newData;
}

export function saveData(data: Data, filePath = DEFAULT_DATA_FILE_PATH): void {
  writeJSONFile(filePath, {...data, redirects: sortRedirects(data.redirects)});
}

export function loadData(filePath = DEFAULT_DATA_FILE_PATH): Data {
  return parseJSONFile(filePath);
}
