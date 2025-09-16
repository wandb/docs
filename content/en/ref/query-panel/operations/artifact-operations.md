---
title: Artifact Operations
---

Operations for querying and manipulating W&B artifacts

## artifactLink

```typescript
artifactLink(artifact): string
```

Gets the URL/link for accessing an artifact in the W&B UI.

Returns a direct link to view the artifact in the W&B web interface,
useful for generating reports or sharing artifact references.

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `artifact` | [`Artifact`](../data-types/artifact.md) | The artifact to get the link for |

#### Example: Generate Artifact Link
```typescript
const link = artifactLink(myArtifact);
console.log(View artifact: ${link});
// Output: https://wandb.ai/entity/project/artifacts/type/name
```

#### Example: Create Markdown Links
```typescript
const artifacts = project.artifacts();
const markdown = artifacts.map(a => 
  - ${artifactName(a)}})
).join('\n');
```

#### See Also

 - [artifactName](#artifactname) - Get artifact name
 - [artifactVersions](#artifactversions) - Get artifact versions

___

## artifactName

```typescript
artifactName(artifact): string
```

Gets the name of an artifact.

Returns the artifact's unique name within its project.

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `artifact` | [`Artifact`](../data-types/artifact.md) | The artifact to get the name from |

#### Example: Display Artifact Names
```typescript
artifacts.forEach(artifact => {
  console.log(Artifact: ${artifactName(artifact)});
});
```

#### Example: Filter by Name Pattern
```typescript
const modelArtifacts = artifacts.filter(a => 
  artifactName(a).includes("model")
);
```

#### See Also

 - [artifactLink](#artifactlink) - Get artifact URL
 - [artifactVersions](#artifactversions) - Get versions

___

## artifactVersionAlias

```typescript
artifactVersionAlias(version): string
```

Gets the alias of an artifact version.

Returns the version alias (e.g., "latest", "best", "production").

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `version` | [`ArtifactVersion`](../data-types/artifactversion.md) | The artifact version |

#### Example: Find Production Version
```typescript
const prodVersion = versions.find(v => 
  artifactVersionAlias(v) === "production"
);
```

#### See Also

[artifactVersions](#artifactversions) - Get all versions

___

## artifactVersionCreatedAt

```typescript
artifactVersionCreatedAt(version): Date
```

Gets the creation date of an artifact version.

Returns when a specific version of the artifact was created.

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `version` | [`ArtifactVersion`](../data-types/artifactversion.md) | The artifact version to get creation date from |

#### Example: Sort Versions by Date
```typescript
const sorted = versions.sort((a, b) => 
  artifactVersionCreatedAt(a).getTime() - 
  artifactVersionCreatedAt(b).getTime()
);
```

#### See Also

[artifactVersions](#artifactversions) - Get all versions

___

## artifactVersionDigest

```typescript
artifactVersionDigest(version): string
```

Gets the content digest/hash of an artifact version.

Returns the SHA256 digest used to verify artifact integrity.

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `version` | [`ArtifactVersion`](../data-types/artifactversion.md) | The artifact version |

#### Example: Verify Artifact Integrity
```typescript
const digest = artifactVersionDigest(version);
const expected = "sha256:abc123...";
if (digest !== expected) {
  console.error("Artifact integrity check failed!");
}
```

#### See Also

[artifactVersions](#artifactversions) - Get all versions

___

## artifactVersionNumber

```typescript
artifactVersionNumber(version): number
```

Gets the version number of an artifact version.

Returns the numeric version identifier.

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `version` | [`ArtifactVersion`](../data-types/artifactversion.md) | The artifact version |

#### Example: Get Latest Version Number
```typescript
const versions = artifactVersions(artifact);
const maxVersion = Math.max(...versions.map(v => 
  artifactVersionNumber(v)
));
console.log(Latest version: v${maxVersion});
```

#### See Also

[artifactVersions](#artifactversions) - Get all versions

___

## artifactVersionSize

```typescript
artifactVersionSize(version): number
```

Gets the size of an artifact version in bytes.

Returns the storage size of a specific artifact version.

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `version` | [`ArtifactVersion`](../data-types/artifactversion.md) | The artifact version to get size from |

#### Example: Display Human-Readable Size
```typescript
const bytes = artifactVersionSize(version);
const mb = (bytes / 1e6).toFixed(2);
console.log(Size: ${mb} MB);
```

#### Example: Find Large Artifacts
```typescript
const largeVersions = versions.filter(v => 
  artifactVersionSize(v) > 1e9 // > 1GB
);
```

#### See Also

[artifactVersions](#artifactversions) - Get all versions

___

## artifactVersions

```typescript
artifactVersions(artifact): ArtifactVersion[]
```

Gets all versions of an artifact.

Returns an array of all version objects for the artifact,
including version numbers, aliases, sizes, and timestamps.

#### Parameters

| Name | Type | Description |
| :------ | :------ | :------ |
| `artifact` | [`Artifact`](../data-types/artifact.md) | The artifact to get versions from |

#### Example: List All Versions
```typescript
const versions = artifactVersions(myArtifact);
versions.forEach(v => {
  console.log(v${v.version}: ${v.alias} (${v.size} bytes));
});
```

#### Example: Find Latest Version
```typescript
const versions = artifactVersions(artifact);
const latest = versions.find(v => v.alias === "latest");
if (latest) {
  console.log(Latest is v${latest.version});
}
```

#### Example: Calculate Total Storage
```typescript
const versions = artifactVersions(artifact);
const totalSize = versions.reduce((sum, v) => sum + v.size, 0);
console.log(Total storage: ${(totalSize / 1e9).toFixed(2)} GB);
```

#### See Also

 - [ArtifactVersion](../data-types/artifactversion.md) - Version type definition
 - [artifactName](#artifactname) - Get artifact name
