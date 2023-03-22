import immer from 'immer';

import type {Redirect} from './lib';
import {parseJSONFile, writeJSONFile} from '../utils';

const DEFAULT_DATA_FILE_PATH = `./data.json`;

type Data = {
  redirects: Redirect[];
  ignoredPaths: Set<string>;
  encounteredPaths: Set<string>;
  suggestionPrefixes: Redirect[];
};

type JSONRepresentation = {
  redirects: Redirect[];
  ignoredPaths: string[];
  suggestionPrefixes: Redirect[];
};

export function addRedirect(data: Data, r: Redirect): Data {
  const newData = immer(data, draft => {
    draft.redirects.push(r);
    draft.encounteredPaths.add(r.from);
  });
  saveData(newData);
  return newData;
}

export function addIgnoredPath(data: Data, path: string): Data {
  const newData = immer(data, draft => {
    draft.ignoredPaths.add(path);
    draft.encounteredPaths.add(path);
  });
  saveData(newData);
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
    redirects: data.redirects,
    ignoredPaths: [...data.ignoredPaths],
    suggestionPrefixes: data.suggestionPrefixes,
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
    suggestionPrefixes: json.suggestionPrefixes,
  };
}
