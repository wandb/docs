import {load, dump} from 'js-yaml';

type ParsedYAML = {
  [key: string]: string;
};

export function updateYAML(
  yamlContent: string,
  key: string,
  value: string
): string {
  const parsedYAML: ParsedYAML = load(yamlContent) as ParsedYAML;
  parsedYAML[key] = value;
  const updatedYAMLContent = dump(parsedYAML).trim();
  return updatedYAMLContent;
}
