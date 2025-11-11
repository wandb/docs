#!/usr/bin/env node

import { compile } from '@mdx-js/mdx';
import { readFile } from 'fs/promises';
import { glob } from 'glob';

async function validateMDX(filePath) {
  try {
    const content = await readFile(filePath, 'utf8');
    await compile(content, { filepath: filePath });
    return { success: true, file: filePath };
  } catch (error) {
    return { 
      success: false, 
      file: filePath, 
      error: error.message,
      line: error.line,
      column: error.column
    };
  }
}

async function main() {
  const files = await glob('**/*.mdx', { 
    ignore: ['node_modules/**', '.git/**'],
    absolute: true
  });
  
  console.log(`Validating ${files.length} MDX files...\n`);
  
  const results = await Promise.all(files.map(validateMDX));
  const failures = results.filter(r => !r.success);
  
  if (failures.length > 0) {
    console.error(`❌ Found ${failures.length} MDX parsing errors:\n`);
    failures.forEach(({ file, error, line, column }) => {
      const relativePath = file.replace(process.cwd() + '/', '');
      console.error(`${relativePath}:${line}:${column}`);
      console.error(`  ${error}\n`);
    });
    process.exit(1);
  } else {
    console.log(`✅ All ${files.length} MDX files are valid!`);
  }
}

main().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});

