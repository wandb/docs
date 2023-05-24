import {readFileSync, writeFileSync} from 'fs';
import yargs from 'yargs';

import {updateFrontMatter} from './frontMatter';
import {findFilesWithExtension} from '../utils';

const {dirPath, sidebarID} = yargs(process.argv.slice(2))
  .options({
    dirPath: {
      alias: 'd',
      type: 'string',
      description: 'Path to the directory',
      require: true,
    },
    sidebarID: {
      alias: 's',
      type: 'string',
      description: 'ID of sidebar to display',
      require: true,
    },
  })
  .parseSync();

(async () => {
  const filePaths = findFilesWithExtension(dirPath, `.md`);
  for (const path of filePaths) {
    updateFile(path);
  }
})();

function updateFile(path: string): void {
  const content = readFileSync(path).toString().trim();
  const newContent = ensureNewlineAtEndOfFile(
    updateFrontMatter(content, `displayed_sidebar`, sidebarID)
  );
  if (content !== newContent) {
    writeFileSync(path, newContent);
  }
}

function ensureNewlineAtEndOfFile(content: string): string {
  if (content.endsWith(`\n`)) {
    return content;
  }
  return content + `\n`;
}
