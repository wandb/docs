import _ from 'lodash';

export const DEFAULT_DATA_FILE_PATH = `./data.json`;

export type Data = {
  redirects: Redirect[];
  ignoredPaths: string[];
};

export type Redirect = {
  from: string;
  to: string;
};

export function sortPaths(paths: string[]): string[] {
  return _.sortBy(paths);
}

export function sortRedirects(redirects: Redirect[]): Redirect[] {
  return _.sortBy(redirects, r => r.from);
}

export function killLeadingSlash(path: string): string {
  if (path.startsWith(`/`)) {
    return path.slice(1);
  }
  return path;
}

export function killTrailingSlash(path: string): string {
  if (path.endsWith(`/`)) {
    return path.slice(0, -1);
  }
  return path;
}

export function isRelativeRedirect(r: Redirect): boolean {
  return !r.from.startsWith(`http`) && !r.to.startsWith(`http`);
}
