# Actual Type Structures from Source Code

Based on inspection of `/scripts/core/frontends/weave/src/core/ops/domain/*.ts`:

## Entity
**Uses `isTeam` boolean, NOT `type` string!**

Available operations show these fields:
- `id` (internal ID, hidden)
- `name`
- `isTeam` (boolean)
- `projects`
- `reports` (hidden)
- `artifactPortfolios` (hidden) 
- `organization` (hidden)
- `artifactTTLDurationSeconds` (hidden)
- `link`

## User
Available operations show these fields:
- `id` (hidden)
- `username`
- `name` (hidden)
- `email` (hidden)  
- `userInfo` (hidden)
- `runs` (hidden)
- `teams`/`entities` (hidden)
- `link` (hidden)

## ArtifactType
Available operations show these fields:
- `name`
- `artifactCollections` (shown as `artifacts`)
- `artifactSequences` (hidden)
- `artifactPortfolios` (hidden)
- `artifactVersions`

## Artifact
Available operations show these fields:
- `id`
- `name`
- `description`
- `aliases`
- `type` (returns artifactType)
- `versions`
- `lastMembership` (hidden)
- `createdAt`
- `project`
- `link`
- `memberships` (hidden)
- `membershipForAlias`
- `isPortfolio` (hidden)
- `rawTags` (hidden)

## ArtifactVersion
Available operations show many fields including:
- `id`
- `version`
- `versionIndex`
- `aliases`
- `artifact`
- `createdAt`
- `createdBy` (can be User)
- `metadata`
- `files`
- `usedBy` (runs)
- `loggedBy` (runs)
- And many more...

## Project
Available operations show these fields:
- `internalId` (hidden)
- `entity`
- `createdAt`
- `updatedAt`
- `name`
- `link`
- `run` (single)
- `runs` (list)
- `artifactType` (single)
- `artifactTypes` (list)
- `artifact` (single)
- `artifacts` (list)
- `artifactVersion` (single)
- `reports` (hidden)
- `runQueues` (hidden)
- `traces` (hidden)

## Run
Available operations show these fields:
- `id`
- `name`
- `project`
- `entity`
- `config` (returns typedDict)
- `summaryMetrics` (returns typedDict)
- `state`
- `createdAt`
- `updatedAt`
- `loggedArtifactVersion` (single)
- `loggedArtifactVersions` (list)
- `usedArtifactVersions` (list)
- Many more...

## Key Findings

1. **Entity uses `isTeam: boolean`**, not `type: "team" | "user"`
2. Most types have many hidden fields not exposed in public API
3. ConfigDict and SummaryDict are actually just `typedDict` - flexible key-value objects
4. Many fields return GraphQL connections that get converted to nodes
