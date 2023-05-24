import {updateYAML} from './yaml';

export function updateFrontMatter(
  content: string,
  key: string,
  value: string
): string {
  // Check if content has front matter
  const frontMatterRegex = /^---\n(.*\n)*?---\n/;
  const frontMatterMatch = content.match(frontMatterRegex);

  if (!frontMatterMatch) {
    return addFrontMatter(content, key, value);
  }

  const frontMatterString = frontMatterMatch[0]!.slice(4, -4).trim();
  const updatedFrontMatterString = updateYAML(frontMatterString, key, value);

  // Replace the original front matter with the updated version
  return content.replace(
    frontMatterRegex,
    `---\n${updatedFrontMatterString}\n---\n`
  );
}

function addFrontMatter(content: string, key: string, value: string): string {
  return `---
${key}: ${value}
---

${content}`;
}
