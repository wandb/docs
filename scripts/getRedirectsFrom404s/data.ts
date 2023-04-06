import immer, {enableMapSet} from 'immer';

import {getSuggestionPrefixes} from './lib';
import {
  Data as JSONRepresentation,
  DEFAULT_DATA_FILE_PATH,
  Redirect,
  parseJSONFile,
  sortPaths,
  sortRedirects,
  writeJSONFile,
} from '../utils';

enableMapSet();

type Data = {
  redirects: Redirect[];
  ignoredPaths: Set<string>;
  encounteredPaths: Set<string>;
  suggestionPrefixes: Redirect[];
};

export function addRedirect(
  data: Data,
  r: Redirect,
  dataFilePath = DEFAULT_DATA_FILE_PATH
): Data {
  const newData = immer(data, draft => {
    draft.redirects.push(r);
    draft.encounteredPaths.add(r.from);
    draft.suggestionPrefixes = getSuggestionPrefixes(draft.redirects);
  });
  saveData(newData, dataFilePath);
  return newData;
}

export function addIgnoredPath(
  data: Data,
  path: string,
  dataFilePath = DEFAULT_DATA_FILE_PATH
): Data {
  const newData = immer(data, draft => {
    draft.ignoredPaths.add(path);
    draft.encounteredPaths.add(path);
  });
  saveData(newData, dataFilePath);
  return newData;
}

export function saveData(data: Data, filePath = DEFAULT_DATA_FILE_PATH): void {
  const json = toJSON(data);
  writeJSONFile(filePath, json);
}

export function loadData(filePath = DEFAULT_DATA_FILE_PATH): Data {
  const json: JSONRepresentation = parseJSONFile(filePath);
  return fromJSON(json);
}

export function toJSON(data: Data): JSONRepresentation {
  return {
    redirects: sortRedirects(data.redirects),
    ignoredPaths: sortPaths([...data.ignoredPaths]),
  };
}

export function fromJSON(json: JSONRepresentation): Data {
  return {
    redirects: json.redirects,
    ignoredPaths: new Set(json.ignoredPaths),
    encounteredPaths: new Set([
      ...json.redirects.map(r => r.from),
      ...json.ignoredPaths,
    ]),
    suggestionPrefixes: getSuggestionPrefixes(json.redirects),
  };
}
