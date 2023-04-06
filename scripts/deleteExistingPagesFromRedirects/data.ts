import {enableMapSet} from 'immer';

import _ from 'lodash';
import {
  DEFAULT_DATA_FILE_PATH,
  Data,
  Redirect,
  parseJSONFile,
  sortRedirects,
  writeJSONFile,
} from '../utils';

enableMapSet();

export function removeRedirects(
  data: Data,
  redirects: Redirect[],
  dataFilePath = DEFAULT_DATA_FILE_PATH
): Data {
  const newData = {
    ...data,
    redirects: _.differenceWith(data.redirects, redirects, _.isEqual),
  };

  saveData(newData, dataFilePath);
  return newData;
}

export function saveData(data: Data, filePath = DEFAULT_DATA_FILE_PATH): void {
  writeJSONFile(filePath, {...data, redirects: sortRedirects(data.redirects)});
}

export function loadData(filePath = DEFAULT_DATA_FILE_PATH): Data {
  return parseJSONFile(filePath);
}
