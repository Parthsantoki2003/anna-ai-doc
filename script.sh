#!/usr/bin/env bash
set -Eeuo pipefail

# Anna Document Workspace - Cloud Run Deployment Script
# Optimized for Google Cloud Run (Stateless, Scaled to Zero, Request-Based Billing)

# ==================== Safe .env Loading ====================
# Strips Windows \r carriage returns so variables are never corrupted on Git Bash / WSL / Cloud Shell
if [[ -f ".env" ]]; then
  echo "Loading configuration from .env..."
  while IFS='=' read -r key value || [[ -n "$key" ]]; do
    [[ "$key" =~ ^[[:space:]]*# ]] && continue
    [[ -z "${key// }" ]] && continue
    clean_key="$(echo "$key" | tr -d '[:space:]')"
    clean_value="$(echo "$value" | tr -d '\r')"
    clean_value="${clean_value#\"}"
    clean_value="${clean_value%\"}"
    clean_value="${clean_value#\'}"
    clean_value="${clean_value%\'}"
    if [[ -n "$clean_key" ]]; then
      export "$clean_key"="$clean_value"
    fi
  done < ".env"
fi

# ==================== Deployment Target & Defaults ====================
PROJECT_ID="${PROJECT_ID:-banded-advice-437316-g8}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="anna-document-workspace"
REPOSITORY="anna-apps"
SERVICE_ACCOUNT_NAME="anna-document-runtime"

QDRANT_URL="${QDRANT_URL:-}"
GROQ_API_KEY="${GROQ_API_KEY:-}"
QDRANT_API_KEY="${QDRANT_API_KEY:-}"
AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-}"
AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-}"
AWS_ENDPOINT_URL_S3="${AWS_ENDPOINT_URL_S3:-}"
AWS_REGION="${AWS_REGION:-ap-southeast-1}"
S3_BUCKET_NAME="${S3_BUCKET_NAME:-documents}"
LLM_MODEL="${LLM_MODEL:-openai/gpt-oss-120b}"
ALLOWED_ORIGINS="${ALLOWED_ORIGINS:-http://localhost:3000,http://localhost:5173,http://localhost:8080,https://anna.partners}"

# Allow passing pre-built image as first argument if available: ./script.sh <image_url>
EXISTING_IMAGE="${1:-}"
IMAGE_TAG="v$(date +%Y%m%d%H%M%S)"

# Basic sanity check
if [[ -z "$GROQ_API_KEY" || -z "$QDRANT_API_KEY" ]]; then
  echo "ERROR: GROQ_API_KEY and QDRANT_API_KEY must be set in .env or the environment."
  exit 1
fi

echo "=========================================================="
echo " Deploying $SERVICE_NAME to GCP Project: $PROJECT_ID"
echo " Region: $REGION"
echo "=========================================================="

echo "1. Selecting GCP project and region..."
gcloud config set project "$PROJECT_ID" --quiet
gcloud config set run/region "$REGION" --quiet

echo "2. Enabling required GCP services..."
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com \
  --quiet

echo "3. Creating / Updating Secret Manager secrets..."
for secret_name in "groq-api-key" "qdrant-api-key" "supabase-s3-access-key" "supabase-s3-secret-key"; do
  case "$secret_name" in
    groq-api-key) secret_val="$GROQ_API_KEY" ;;
    qdrant-api-key) secret_val="$QDRANT_API_KEY" ;;
    supabase-s3-access-key) secret_val="$AWS_ACCESS_KEY_ID" ;;
    supabase-s3-secret-key) secret_val="$AWS_SECRET_ACCESS_KEY" ;;
  esac

  if ! gcloud secrets describe "$secret_name" --project="$PROJECT_ID" >/dev/null 2>&1; then
    printf '%s' "$secret_val" | gcloud secrets create "$secret_name" \
      --project="$PROJECT_ID" \
      --replication-policy="automatic" \
      --data-file=- \
      --quiet
  else
    printf '%s' "$secret_val" | gcloud secrets versions add "$secret_name" \
      --project="$PROJECT_ID" \
      --data-file=- \
      --quiet
  fi
done

echo "4. Setting up runtime service account..."
SERVICE_ACCOUNT="$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com"
if ! gcloud iam service-accounts describe "$SERVICE_ACCOUNT" --project="$PROJECT_ID" >/dev/null 2>&1; then
  gcloud iam service-accounts create "$SERVICE_ACCOUNT_NAME" \
    --project="$PROJECT_ID" \
    --display-name="Anna Document Runtime" \
    --quiet
fi

echo "5. Granting Secret Manager access to service account..."
for secret_name in "groq-api-key" "qdrant-api-key" "supabase-s3-access-key" "supabase-s3-secret-key"; do
  gcloud secrets add-iam-policy-binding "$secret_name" \
    --project="$PROJECT_ID" \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/secretmanager.secretAccessor" \
    --quiet >/dev/null 2>&1 || true
done

echo "6. Ensuring Artifact Registry repository exists..."
if ! gcloud artifacts repositories describe "$REPOSITORY" --location="$REGION" --project="$PROJECT_ID" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$REPOSITORY" \
    --repository-format=docker \
    --location="$REGION" \
    --project="$PROJECT_ID" \
    --description="Anna Document Workspace Docker repository" \
    --quiet
fi

# Ensure Cloud Build has permission to push to Artifact Registry
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
gcloud artifacts repositories add-iam-policy-binding "$REPOSITORY" \
  --location="$REGION" \
  --project="$PROJECT_ID" \
  --member="serviceAccount:$PROJECT_NUMBER@cloudbuild.gserviceaccount.com" \
  --role="roles/artifactregistry.writer" \
  --quiet >/dev/null 2>&1 || true

if [[ -n "$EXISTING_IMAGE" ]]; then
  IMAGE="$EXISTING_IMAGE"
  echo "Using specified image: $IMAGE"
else
  IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$SERVICE_NAME:$IMAGE_TAG"
  echo "7. Building and submitting Docker image via Cloud Build..."
  gcloud builds submit \
    --tag "$IMAGE" \
    .
fi

echo "8. Writing env.yaml to avoid comma-parsing errors..."
cat <<EOF > env.yaml
QDRANT_URL: "$QDRANT_URL"
ALLOWED_ORIGINS: "$ALLOWED_ORIGINS"
WORKERS: "1"
LLM_MODEL: "$LLM_MODEL"
LOG_LEVEL: "INFO"
ANNA_APP_SLUG: "anna-document-workspace"
ANNA_APP_VERSION: "1.0.0"
ANNA_APP_MODE: "cloud-compatible"
AWS_ENDPOINT_URL_S3: "$AWS_ENDPOINT_URL_S3"
AWS_REGION: "$AWS_REGION"
S3_BUCKET_NAME: "$S3_BUCKET_NAME"
EOF

echo "9. Deploying to Cloud Run with Cost & Concurrency Optimizations..."
gcloud run deploy "$SERVICE_NAME" \
  --image "$IMAGE" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --service-account "$SERVICE_ACCOUNT" \
  --memory 1536Mi \
  --cpu 1 \
  --concurrency 80 \
  --min-instances 0 \
  --max-instances 5 \
  --timeout 300 \
  --cpu-throttling \
  --env-vars-file="env.yaml" \
  --set-secrets "GROQ_API_KEY=groq-api-key:latest,QDRANT_API_KEY=qdrant-api-key:latest,AWS_ACCESS_KEY_ID=supabase-s3-access-key:latest,AWS_SECRET_ACCESS_KEY=supabase-s3-secret-key:latest" \
  --quiet

rm -f env.yaml

echo
echo "=========================================================="
echo " Deployment SUCCESSFUL!"
echo " Service URL:"
gcloud run services describe "$SERVICE_NAME" \
  --region "$REGION" \
  --project="$PROJECT_ID" \
  --format='value(status.url)'
echo "=========================================================="
