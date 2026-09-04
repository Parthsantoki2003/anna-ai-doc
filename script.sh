#!/usr/bin/env bash
set -Eeuo pipefail

# Anna Document Workspace - simple GCP deployment
# Run this from Google Cloud Shell or Git Bash after replacing the values below.

# ==================== REQUIRED VALUES ====================
if [[ -f ".env" ]]; then
  set -a
  source ".env"
  set +a
fi

PROJECT_ID="${PROJECT_ID:-banded-advice-437316-g8}"
REGION="${REGION:-us-central1}"
QDRANT_URL="${QDRANT_URL:-https://YOUR_QDRANT_CLUSTER_URL}"
GROQ_API_KEY="${GROQ_API_KEY:-YOUR_GROQ_API_KEY}"
QDRANT_API_KEY="${QDRANT_API_KEY:-YOUR_QDRANT_API_KEY}"
AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-YOUR_SUPABASE_S3_ACCESS_KEY}"
AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-YOUR_SUPABASE_S3_SECRET_KEY}"
AWS_ENDPOINT_URL_S3="${AWS_ENDPOINT_URL_S3:-https://YOUR_PROJECT_REF.storage.supabase.co/storage/v1/s3}"
AWS_REGION="${AWS_REGION:-us-east-1}"
S3_BUCKET_NAME="${S3_BUCKET_NAME:-documents}"

# ==================== App settings ====================
SERVICE_NAME="anna-document-workspace"
REPOSITORY="anna-apps"
IMAGE_TAG="1.0.0"
SERVICE_ACCOUNT_NAME="anna-document-runtime"
LLM_MODEL="openai/gpt-oss-120b"
ALLOWED_ORIGINS="*"

if [[ "$PROJECT_ID" == "YOUR_GCP_PROJECT_ID" ||
      "$QDRANT_URL" == "https://YOUR_QDRANT_CLUSTER_URL" ||
      "$GROQ_API_KEY" == "YOUR_GROQ_API_KEY" ||
      "$QDRANT_API_KEY" == "YOUR_QDRANT_API_KEY" ||
      "$AWS_ACCESS_KEY_ID" == "YOUR_SUPABASE_S3_ACCESS_KEY" ||
      "$AWS_SECRET_ACCESS_KEY" == "YOUR_SUPABASE_S3_SECRET_KEY" ]]; then
  echo "Edit the required placeholder values at the top of script.sh first."
  exit 1
fi

echo "Selecting GCP project..."
gcloud config set project "$PROJECT_ID"
gcloud config set run/region "$REGION"

echo "Enabling required GCP services..."
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com

echo "Creating Secret Manager secrets..."
if ! gcloud secrets describe groq-api-key >/dev/null 2>&1; then
  printf '%s' "$GROQ_API_KEY" | gcloud secrets create groq-api-key --data-file=-
else
  printf '%s' "$GROQ_API_KEY" | gcloud secrets versions add groq-api-key --data-file=-
fi

if ! gcloud secrets describe qdrant-api-key >/dev/null 2>&1; then
  printf '%s' "$QDRANT_API_KEY" | gcloud secrets create qdrant-api-key --data-file=-
else
  printf '%s' "$QDRANT_API_KEY" | gcloud secrets versions add qdrant-api-key --data-file=-
fi

if ! gcloud secrets describe supabase-s3-access-key >/dev/null 2>&1; then
  printf '%s' "$AWS_ACCESS_KEY_ID" | gcloud secrets create supabase-s3-access-key --data-file=-
else
  printf '%s' "$AWS_ACCESS_KEY_ID" | gcloud secrets versions add supabase-s3-access-key --data-file=-
fi

if ! gcloud secrets describe supabase-s3-secret-key >/dev/null 2>&1; then
  printf '%s' "$AWS_SECRET_ACCESS_KEY" | gcloud secrets create supabase-s3-secret-key --data-file=-
else
  printf '%s' "$AWS_SECRET_ACCESS_KEY" | gcloud secrets versions add supabase-s3-secret-key --data-file=-
fi

echo "Creating runtime service account..."
if ! gcloud iam service-accounts describe \
  "$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com" >/dev/null 2>&1; then
  gcloud iam service-accounts create "$SERVICE_ACCOUNT_NAME" \
    --display-name="Anna Document Runtime"
fi

SERVICE_ACCOUNT="$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com"

echo "Granting Secret Manager access..."
gcloud secrets add-iam-policy-binding groq-api-key \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/secretmanager.secretAccessor"
gcloud secrets add-iam-policy-binding qdrant-api-key \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/secretmanager.secretAccessor"
gcloud secrets add-iam-policy-binding supabase-s3-access-key \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/secretmanager.secretAccessor"
gcloud secrets add-iam-policy-binding supabase-s3-secret-key \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/secretmanager.secretAccessor"

echo "Creating Artifact Registry repository..."
if ! gcloud artifacts repositories describe "$REPOSITORY" \
  --location="$REGION" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$REPOSITORY" \
    --repository-format=docker \
    --location="$REGION"
fi

gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet

IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$SERVICE_NAME:$IMAGE_TAG"

echo "Building and uploading Docker image..."
gcloud builds submit \
  --tag "$IMAGE" \
  .

echo "Deploying to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
  --image "$IMAGE" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --service-account "$SERVICE_ACCOUNT" \
  --memory 4Gi \
  --cpu 2 \
  --concurrency 1 \
  --min-instances 0 \
  --max-instances 3 \
  --timeout 900 \
  --set-env-vars "QDRANT_URL=$QDRANT_URL,ALLOWED_ORIGINS=$ALLOWED_ORIGINS,WORKERS=1,LLM_MODEL=$LLM_MODEL,LOG_LEVEL=INFO,ANNA_APP_SLUG=anna-document-workspace,ANNA_APP_VERSION=1.0.0,ANNA_APP_MODE=cloud-compatible,AWS_ENDPOINT_URL_S3=$AWS_ENDPOINT_URL_S3,AWS_REGION=$AWS_REGION,S3_BUCKET_NAME=$S3_BUCKET_NAME" \
  --set-secrets "GROQ_API_KEY=groq-api-key:latest,QDRANT_API_KEY=qdrant-api-key:latest,AWS_ACCESS_KEY_ID=supabase-s3-access-key:latest,AWS_SECRET_ACCESS_KEY=supabase-s3-secret-key:latest"

echo
echo "Deployment complete."
echo "Backend URL:"
gcloud run services describe "$SERVICE_NAME" \
  --region "$REGION" \
  --format='value(status.url)'
