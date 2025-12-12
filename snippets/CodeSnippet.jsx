/**
 * CodeSnippet component for including Python code examples by filename
 * Usage: <CodeSnippet file="artifact_create.py" />
 * 
 * This component consolidates all code example imports in one place,
 * allowing cheat sheet pages to reference snippets by filename without
 * needing individual import statements.
 * 
 * AUTO-GENERATED: Do not edit manually. Run sync_code_examples.sh to regenerate.
 */

// Import all Python code examples as raw text
import artifact_add_alias from './en/_includes/code-examples/artifact_add_alias.py?raw';
import artifact_add_alias_existing from './en/_includes/code-examples/artifact_add_alias_existing.py?raw';
import artifact_add_tag from './en/_includes/code-examples/artifact_add_tag.py?raw';
import artifact_add_tag_existing from './en/_includes/code-examples/artifact_add_tag_existing.py?raw';
import artifact_create from './en/_includes/code-examples/artifact_create.py?raw';
import artifact_create_track_external from './en/_includes/code-examples/artifact_create_track_external.py?raw';
import artifact_delete from './en/_includes/code-examples/artifact_delete.py?raw';
import artifact_delete_collection from './en/_includes/code-examples/artifact_delete_collection.py?raw';
import artifact_download_partial from './en/_includes/code-examples/artifact_download_partial.py?raw';
import artifact_ttl from './en/_includes/code-examples/artifact_ttl.py?raw';
import artifact_ttl_existing from './en/_includes/code-examples/artifact_ttl_existing.py?raw';
import artifact_update from './en/_includes/code-examples/artifact_update.py?raw';
import artifact_update_existing from './en/_includes/code-examples/artifact_update_existing.py?raw';
import artifacts_download from './en/_includes/code-examples/artifacts_download.py?raw';
import experiments_create from './en/_includes/code-examples/experiments_create.py?raw';
import log_custom_summary_metric from './en/_includes/code-examples/log_custom_summary_metric.py?raw';
import log_existing_artifact_link_collection from './en/_includes/code-examples/log_existing_artifact_link_collection.py?raw';
import log_hyperparameter from './en/_includes/code-examples/log_hyperparameter.py?raw';
import log_metric from './en/_includes/code-examples/log_metric.py?raw';
import log_table from './en/_includes/code-examples/log_table.py?raw';
import registry_add_annotation from './en/_includes/code-examples/registry_add_annotation.py?raw';
import registry_collection_create from './en/_includes/code-examples/registry_collection_create.py?raw';
import registry_collection_tags_add from './en/_includes/code-examples/registry_collection_tags_add.py?raw';
import registry_collection_tags_remove from './en/_includes/code-examples/registry_collection_tags_remove.py?raw';
import registry_create from './en/_includes/code-examples/registry_create.py?raw';
import registry_delete from './en/_includes/code-examples/registry_delete.py?raw';
import registry_link_artifact_existing from './en/_includes/code-examples/registry_link_artifact_existing.py?raw';
import registry_use_linked_artifact from './en/_includes/code-examples/registry_use_linked_artifact.py?raw';
import run_fork from './en/_includes/code-examples/run_fork.py?raw';
import run_init from './en/_includes/code-examples/run_init.py?raw';

// Map filenames to imported content
const snippets = {
  'artifact_add_alias.py': artifact_add_alias,
  'artifact_add_alias_existing.py': artifact_add_alias_existing,
  'artifact_add_tag.py': artifact_add_tag,
  'artifact_add_tag_existing.py': artifact_add_tag_existing,
  'artifact_create.py': artifact_create,
  'artifact_create_track_external.py': artifact_create_track_external,
  'artifact_delete.py': artifact_delete,
  'artifact_delete_collection.py': artifact_delete_collection,
  'artifact_download_partial.py': artifact_download_partial,
  'artifact_ttl.py': artifact_ttl,
  'artifact_ttl_existing.py': artifact_ttl_existing,
  'artifact_update.py': artifact_update,
  'artifact_update_existing.py': artifact_update_existing,
  'artifacts_download.py': artifacts_download,
  'experiments_create.py': experiments_create,
  'log_custom_summary_metric.py': log_custom_summary_metric,
  'log_existing_artifact_link_collection.py': log_existing_artifact_link_collection,
  'log_hyperparameter.py': log_hyperparameter,
  'log_metric.py': log_metric,
  'log_table.py': log_table,
  'registry_add_annotation.py': registry_add_annotation,
  'registry_collection_create.py': registry_collection_create,
  'registry_collection_tags_add.py': registry_collection_tags_add,
  'registry_collection_tags_remove.py': registry_collection_tags_remove,
  'registry_create.py': registry_create,
  'registry_delete.py': registry_delete,
  'registry_link_artifact_existing.py': registry_link_artifact_existing,
  'registry_use_linked_artifact.py': registry_use_linked_artifact,
  'run_fork.py': run_fork,
  'run_init.py': run_init,
};

export const CodeSnippet = ({ file }) => {
  const content = snippets[file];
  
  if (!content) {
    return (
      <div style={ padding: '1rem', background: '#fee', border: '1px solid #fcc', borderRadius: '4px' }>
        <p style={ margin: 0, color: '#c00' }>Code snippet not found: {file}</p>
      </div>
    );
  }
  
  return (
    <pre>
      <code className="language-python">{content}</code>
    </pre>
  );
};
