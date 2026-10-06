#!/usr/bin/env bash
# Deploy the agent to Azure Container Apps. Run it again to deploy a new version.
#
# Needs: az login, Docker, the search index in data/chroma (scripts/build_index.py),
# and in .env: AZURE_RESOURCE_GROUP, AZURE_LOCATION, AZURE_ACR, AZURE_OPENAI_RESOURCE,
# AZURE_OPENAI_DEPLOYMENT, AZURE_APP_API_KEY (the password for /ask on Azure).
set -euo pipefail
set -a; source .env; set +a

APP=energiewende-agent
ENVIRONMENT=energiewende-env
IMAGE="$AZURE_ACR.azurecr.io/energiewende-agent"
TAG=$(git rev-parse --short HEAD)

# 1. A container registry for our images (Basic is the cheapest).
if ! az acr show -n "$AZURE_ACR" -o none 2>/dev/null; then
  az acr create -n "$AZURE_ACR" -g "$AZURE_RESOURCE_GROUP" -l "$AZURE_LOCATION" --sku Basic --admin-enabled true -o none
fi
az acr login -n "$AZURE_ACR"

# 2. Build for Azure's CPUs (linux/amd64) and push: the app image, then the app image plus the search index.
docker buildx build --platform linux/amd64 -t "$IMAGE:base-$TAG" --push .
docker buildx build --platform linux/amd64 -f Dockerfile.azure --build-arg BASE="$IMAGE:base-$TAG" -t "$IMAGE:$TAG" --push .

# 3. Read the LLM key and the registry password from Azure (they are not stored in .env).
LLM_URL="https://$AZURE_OPENAI_RESOURCE.openai.azure.com/openai/v1/"
LLM_KEY=$(az cognitiveservices account keys list -n "$AZURE_OPENAI_RESOURCE" -g "$AZURE_RESOURCE_GROUP" --query key1 -o tsv)
ACR_PASSWORD=$(az acr credential show -n "$AZURE_ACR" --query "passwords[0].value" -o tsv)

# 4. The Container Apps environment (once), then create or update the app.
if ! az containerapp env show -n "$ENVIRONMENT" -g "$AZURE_RESOURCE_GROUP" -o none 2>/dev/null; then
  az containerapp env create -n "$ENVIRONMENT" -g "$AZURE_RESOURCE_GROUP" -l "$AZURE_LOCATION" -o none
fi

if az containerapp show -n "$APP" -g "$AZURE_RESOURCE_GROUP" -o none 2>/dev/null; then
  az containerapp update -n "$APP" -g "$AZURE_RESOURCE_GROUP" --image "$IMAGE:$TAG" -o none
else
  # min-replicas 0: no costs while nobody uses it (the first request then takes longer).
  az containerapp create -n "$APP" -g "$AZURE_RESOURCE_GROUP" --environment "$ENVIRONMENT" \
    --image "$IMAGE:$TAG" \
    --registry-server "$AZURE_ACR.azurecr.io" --registry-username "$AZURE_ACR" --registry-password "$ACR_PASSWORD" \
    --target-port 8000 --ingress external \
    --cpu 1.5 --memory 3Gi --min-replicas 0 --max-replicas 1 \
    --secrets llm-api-key="$LLM_KEY" app-api-key="$AZURE_APP_API_KEY" \
    --env-vars LLM_BASE_URL="$LLM_URL" LLM_MODEL="$AZURE_OPENAI_DEPLOYMENT" \
               LLM_API_KEY=secretref:llm-api-key APP_API_KEY=secretref:app-api-key \
    -o none
fi

echo "Deployed: https://$(az containerapp show -n "$APP" -g "$AZURE_RESOURCE_GROUP" --query properties.configuration.ingress.fqdn -o tsv)"
