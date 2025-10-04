"""
W&B Automations API - Prerequisites and Examples

This script demonstrates how to check for team-level prerequisites (Slack integrations
and webhooks) that must be configured through the UI before creating automations
programmatically.

IMPORTANT: Team-level integrations CANNOT be created via API for security reasons:
- Slack integrations require OAuth authentication flow
- Webhooks require secure storage of endpoints and credentials
"""

import wandb
from wandb.automations import (
    OnRunMetric, RunEvent,
    OnCreateArtifact, OnAddArtifactAlias,
    SendNotification, SendWebhook
)


def check_slack_prerequisites(api, entity):
    """Check if Slack integrations are configured for the team"""
    print(f"\n=== Checking Slack Integrations for {entity} ===")
    
    slack_integrations = list(api.slack_integrations(entity=entity))
    
    if not slack_integrations:
        print("❌ No Slack integrations found!")
        print(f"\nTo create Slack automations, a team admin must:")
        print(f"1. Go to: https://wandb.ai/{entity}/settings/integrations")
        print(f"2. Click 'Connect Slack'")
        print(f"3. Sign in to Slack and select a channel")
        print(f"4. Grant W&B permission to post to the channel")
        print(f"\nThis CANNOT be done via API due to Slack OAuth requirements.")
        return None
    
    print(f"✅ Found {len(slack_integrations)} Slack integration(s):\n")
    for i, integration in enumerate(slack_integrations):
        print(f"{i+1}. Channel: #{integration.channel_name}")
        print(f"   Slack Team: {integration.slack_team_name}")
        print(f"   Integration ID: {integration.id}")
        print()
    
    return slack_integrations


def check_webhook_prerequisites(api, entity):
    """Check if webhook integrations are configured for the team"""
    print(f"\n=== Checking Webhook Integrations for {entity} ===")
    
    webhook_integrations = list(api.webhook_integrations(entity=entity))
    
    if not webhook_integrations:
        print("❌ No webhook integrations found!")
        print(f"\nTo create webhook automations, a team admin must:")
        print(f"1. Go to: https://wandb.ai/{entity}/settings/webhooks")
        print(f"2. Click 'New webhook'")
        print(f"3. Configure:")
        print(f"   - Webhook name")
        print(f"   - Endpoint URL")
        print(f"   - Access token (optional)")
        print(f"   - Associated secrets (optional)")
        print(f"4. Test the webhook configuration")
        print(f"\nIf your webhook needs secrets:")
        print(f"1. Go to: https://wandb.ai/{entity}/settings/secrets")
        print(f"2. Create secrets for sensitive values")
        print(f"3. Associate them with the webhook")
        print(f"\nThis CANNOT be done via API for security reasons.")
        return None
    
    print(f"✅ Found {len(webhook_integrations)} webhook integration(s):\n")
    for i, webhook in enumerate(webhook_integrations):
        print(f"{i+1}. Name: {webhook.name}")
        print(f"   URL: {webhook.url_endpoint}")
        print(f"   Integration ID: {webhook.id}")
        if hasattr(webhook, 'has_access_token') and webhook.has_access_token:
            print(f"   Auth: Has access token (use ${ACCESS_TOKEN} in payload)")
        print()
    
    return webhook_integrations


def get_slack_integration(api, entity, channel_pattern=None):
    """Get a Slack integration, optionally filtered by channel name"""
    integrations = list(api.slack_integrations(entity=entity))
    
    if not integrations:
        raise ValueError(
            f"No Slack integrations found for {entity}. "
            f"Please configure one at: https://wandb.ai/{entity}/settings/integrations"
        )
    
    if channel_pattern:
        matching = [i for i in integrations if channel_pattern in i.channel_name]
        if matching:
            print(f"Found Slack channel matching '{channel_pattern}': {matching[0].channel_name}")
            return matching[0]
        else:
            print(f"No channel matching '{channel_pattern}', using: {integrations[0].channel_name}")
    
    return integrations[0]


def get_webhook_integration(api, entity, name_pattern=None, url_pattern=None):
    """Get a webhook integration by name or URL pattern"""
    webhooks = list(api.webhook_integrations(entity=entity))
    
    if not webhooks:
        raise ValueError(
            f"No webhook integrations found for {entity}. "
            f"Please configure one at: https://wandb.ai/{entity}/settings/webhooks"
        )
    
    if name_pattern:
        matching = [w for w in webhooks if name_pattern in w.name]
        if matching:
            print(f"Found webhook matching name '{name_pattern}': {matching[0].name}")
            return matching[0]
    
    if url_pattern:
        matching = [w for w in webhooks if url_pattern in w.url_endpoint]
        if matching:
            print(f"Found webhook matching URL '{url_pattern}': {matching[0].name}")
            return matching[0]
    
    print(f"Using first available webhook: {webhooks[0].name}")
    return webhooks[0]


def create_slack_automation_example(api, entity, project_name):
    """Example: Create a Slack automation for low accuracy alerts"""
    try:
        # Get Slack integration
        slack_integration = get_slack_integration(api, entity, "alerts")
        
        # Get project
        project = api.project(project_name, entity=entity)
        
        # Define event
        event = OnRunMetric(
            scope=project,
            filter=RunEvent.metric("accuracy") < 0.85
        )
        
        # Define action
        action = SendNotification.from_integration(
            slack_integration,
            title="⚠️ Low Accuracy Alert",
            text=f"Model accuracy dropped below 85% in {project_name}",
            level="WARN"
        )
        
        # Create automation
        automation = api.create_automation(
            event >> action,
            name="low-accuracy-alert-example",
            description="Alert when model accuracy drops below threshold"
        )
        
        print(f"\n✅ Created Slack automation: {automation.name}")
        print(f"   Will post to: #{slack_integration.channel_name}")
        print(f"   Trigger: accuracy < 0.85")
        
    except Exception as e:
        print(f"\n❌ Failed to create Slack automation: {e}")


def create_webhook_automation_example(api, entity, project_name):
    """Example: Create a webhook automation for model deployment"""
    try:
        # Get webhook integration (prefer GitHub if available)
        webhook = get_webhook_integration(api, entity, url_pattern="github.com")
        
        # Get project
        project = api.project(project_name, entity=entity)
        
        # Define event
        event = OnRunMetric(
            scope=project,
            filter=(RunEvent.metric("val_accuracy") > 0.95) & 
                   (RunEvent.metric("val_loss") < 0.05)
        )
        
        # Define payload
        payload = {
            "event_type": "DEPLOY_MODEL",
            "client_payload": {
                "project": "${project_name}",
                "entity": "${entity_name}",
                "author": "${event_author}",
                "artifact_version": "${artifact_version_string}",
                "metrics": {
                    "event_type": "${event_type}"
                }
            }
        }
        
        # Note about authentication
        if hasattr(webhook, 'has_access_token') and webhook.has_access_token:
            print(f"\n📌 Note: This webhook has an access token configured.")
            print(f"   W&B will automatically add it as Authorization header.")
        
        # Create action
        action = SendWebhook.from_integration(webhook, payload=payload)
        
        # Create automation
        automation = api.create_automation(
            event >> action,
            name="high-performance-deployment-example",
            description="Trigger deployment for high-performing models"
        )
        
        print(f"\n✅ Created webhook automation: {automation.name}")
        print(f"   Will POST to: {webhook.url_endpoint}")
        print(f"   Trigger: val_accuracy > 0.95 AND val_loss < 0.05")
        
    except Exception as e:
        print(f"\n❌ Failed to create webhook automation: {e}")


def main():
    """Main function to demonstrate prerequisites and automation creation"""
    
    # Configuration
    ENTITY = "your-team"  # Replace with your team name
    PROJECT = "your-project"  # Replace with your project name
    
    print("W&B Automations API - Prerequisites Check")
    print("=" * 50)
    
    # Authenticate
    wandb.login()
    api = wandb.Api()
    
    # Check prerequisites
    slack_integrations = check_slack_prerequisites(api, ENTITY)
    webhook_integrations = check_webhook_prerequisites(api, ENTITY)
    
    # Summary
    print("\n=== Summary ===")
    print(f"Entity: {ENTITY}")
    print(f"Slack integrations available: {'✅ Yes' if slack_integrations else '❌ No'}")
    print(f"Webhook integrations available: {'✅ Yes' if webhook_integrations else '❌ No'}")
    
    # Create example automations if prerequisites are met
    if slack_integrations:
        print("\n" + "=" * 50)
        print("Creating example Slack automation...")
        create_slack_automation_example(api, ENTITY, PROJECT)
    
    if webhook_integrations:
        print("\n" + "=" * 50)
        print("Creating example webhook automation...")
        create_webhook_automation_example(api, ENTITY, PROJECT)
    
    # Final notes
    print("\n" + "=" * 50)
    print("IMPORTANT NOTES:")
    print("1. Team-level integrations (Slack, webhooks) MUST be created in the UI")
    print("2. Once configured, you can create unlimited automations via API")
    print("3. This separation ensures security while enabling automation at scale")
    print("\nFor more examples, see the documentation at:")
    print("https://docs.wandb.ai/guides/automations/")


if __name__ == "__main__":
    main()