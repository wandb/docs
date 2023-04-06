import {enableMapSet} from 'immer';

import {DEFAULT_DATA_FILE_PATH, Data, parseJSONFile} from '../utils';

enableMapSet();

export function loadData(filePath = DEFAULT_DATA_FILE_PATH): Data {
  return parseJSONFile(filePath);
}
