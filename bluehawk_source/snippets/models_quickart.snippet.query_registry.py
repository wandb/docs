
# Initialize wandb API
api = wandb.Api()

# Find all artifact versions that contains the string `model` and 
# has either the tag `text-classification` or an `latest` alias
registry_filters = {
    "name": {"$regex": "model"}
}

# Use logical $or operator to filter artifact versions
version_filters = {
    "$or": [
        {"tag": "text-classification"},
        {"alias": "latest"}
    ]
}

# Returns an iterable of all artifact versions that match the filters
artifacts = api.registries(filter=registry_filters).collections().versions(filter=version_filters)

# Print out the name, collection, aliases, tags, and created_at date of each artifact found
for art in artifacts:
    print(f"artifact name: {art.name}")
    print(f"collection artifact belongs to: { art.collection.name}")
    print(f"artifact aliases: {art.aliases}")
    print(f"tags attached to artifact: {art.tags}")
    print(f"artifact created at: {art.created_at}\n")

