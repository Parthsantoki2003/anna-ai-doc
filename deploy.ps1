<#
.SYNOPSIS
    Deploys Anna Document Workspace to Google Cloud Run natively from Windows PowerShell.
#>

param (
    [string]$ExistingImage = ""
)

$ErrorActionPreference = "Continue"

# Load environment variables from .env
if (Test-Path ".env") {
    Write-Host "Loading configuration from .env..." -ForegroundColor Cyan
    Get-Content ".env" | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#")) {
            $parts = $line.Split("=", 2)
            if ($parts.Length -eq 2) {
                $name = $parts[0].Trim()
                $val = $parts[1].Trim().Trim('"').Trim("'")
                [System.Environment]::SetEnvironmentVariable($name, $val, "Process")
            }
        }
    }
}

$ProjectId = if ($env:PROJECT_ID) { $env:PROJECT_ID } else { "banded-advice-437316-g8" }
$Region = if ($env:REGION) { $env:REGION } else { "us-central1" }
$ServiceName = "anna-document-workspace"
$Repository = "anna-apps"
$ServiceAccountName = "anna-document-runtime"
$ServiceAccount = "$ServiceAccountName@$ProjectId.iam.gserviceaccount.com"

$QdrantUrl = $env:QDRANT_URL
$GroqApiKey = $env:GROQ_API_KEY
$QdrantApiKey = $env:QDRANT_API_KEY
$AwsAccessKey = $env:AWS_ACCESS_KEY_ID
$AwsSecretKey = $env:AWS_SECRET_ACCESS_KEY
$AwsEndpoint = $env:AWS_ENDPOINT_URL_S3
$AwsRegion = if ($env:AWS_REGION) { $env:AWS_REGION } else { "ap-southeast-1" }
$S3Bucket = if ($env:S3_BUCKET_NAME) { $env:S3_BUCKET_NAME } else { "documents" }
$LlmModel = if ($env:LLM_MODEL) { $env:LLM_MODEL } else { "openai/gpt-oss-120b" }
$AllowedOrigins = if ($env:ALLOWED_ORIGINS) { $env:ALLOWED_ORIGINS } else { "http://localhost:3000,http://localhost:5173,http://localhost:8080,https://anna.partners" }

if (-not $GroqApiKey -or -not $QdrantApiKey) {
    Write-Error "GROQ_API_KEY and QDRANT_API_KEY must be defined in .env or the current environment."
}

if ($ExistingImage) {
    $Image = $ExistingImage
    Write-Host "Using pre-built image: $Image" -ForegroundColor Cyan
} else {
    $Timestamp = Get-Date -Format "yyyyMMddHHmmss"
    $ImageTag = "v$Timestamp"
    $Image = "$Region-docker.pkg.dev/$ProjectId/$Repository/${ServiceName}:$ImageTag"
}

Write-Host "==========================================================" -ForegroundColor Green
Write-Host " Deploying $ServiceName to GCP Project: $ProjectId" -ForegroundColor Green
Write-Host " Region: $Region | Image: $Image" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

Write-Host "1. Setting GCP project and region..." -ForegroundColor Yellow
gcloud config set project "$ProjectId" --quiet
gcloud config set run/region "$Region" --quiet

Write-Host "2. Enabling required GCP services..." -ForegroundColor Yellow
gcloud services enable `
  run.googleapis.com `
  artifactregistry.googleapis.com `
  cloudbuild.googleapis.com `
  secretmanager.googleapis.com `
  --quiet

Write-Host "3. Creating / Updating Secret Manager secrets..." -ForegroundColor Yellow
$secrets = @{
    "groq-api-key" = $GroqApiKey
    "qdrant-api-key" = $QdrantApiKey
    "supabase-s3-access-key" = $AwsAccessKey
    "supabase-s3-secret-key" = $AwsSecretKey
}

foreach ($name in $secrets.Keys) {
    $val = ("" + $secrets[$name]).Trim()
    if (-not $val) { continue }
    $tempSecretFile = [System.IO.Path]::GetTempFileName()
    try {
        [System.IO.File]::WriteAllText($tempSecretFile, $val, (New-Object System.Text.UTF8Encoding($false)))
        $describe = gcloud secrets describe $name --project="$ProjectId" 2>&1
        if ($LASTEXITCODE -ne 0) {
            gcloud secrets create $name --project="$ProjectId" --replication-policy="automatic" --data-file="$tempSecretFile" --quiet
        } else {
            gcloud secrets versions add $name --project="$ProjectId" --data-file="$tempSecretFile" --quiet
        }
    } finally {
        if (Test-Path $tempSecretFile) {
            Remove-Item $tempSecretFile -Force -ErrorAction SilentlyContinue
        }
    }
}

Write-Host "4. Setting up runtime service account..." -ForegroundColor Yellow
$saDescribe = gcloud iam service-accounts describe $ServiceAccount --project="$ProjectId" 2>&1
if ($LASTEXITCODE -ne 0) {
    gcloud iam service-accounts create $ServiceAccountName --project="$ProjectId" --display-name="Anna Document Runtime" --quiet
}

Write-Host "5. Granting Secret Manager permissions to service account..." -ForegroundColor Yellow
foreach ($name in $secrets.Keys) {
    gcloud secrets add-iam-policy-binding $name `
      --project="$ProjectId" `
      --member="serviceAccount:$ServiceAccount" `
      --role="roles/secretmanager.secretAccessor" `
      --quiet *>$null
}

Write-Host "6. Ensuring Artifact Registry repository exists..." -ForegroundColor Yellow
$repoDescribe = gcloud artifacts repositories describe $Repository --location="$Region" --project="$ProjectId" 2>&1
if ($LASTEXITCODE -ne 0) {
    gcloud artifacts repositories create $Repository `
      --repository-format=docker `
      --location="$Region" `
      --project="$ProjectId" `
      --description="Anna Document Workspace Docker repository" `
      --quiet
}

$ProjectNumber = (gcloud projects describe "$ProjectId" --format='value(projectNumber)').Trim()
gcloud artifacts repositories add-iam-policy-binding $Repository `
  --location="$Region" `
  --project="$ProjectId" `
  --member="serviceAccount:$ProjectNumber@cloudbuild.gserviceaccount.com" `
  --role="roles/artifactregistry.writer" `
  --quiet *>$null

if (-not $ExistingImage) {
    Write-Host "7. Building Docker image with Cloud Build..." -ForegroundColor Yellow
    gcloud builds submit --tag "$Image" .
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Cloud Build failed with exit code $LASTEXITCODE."
        exit $LASTEXITCODE
    }
} else {
    Write-Host "7. Skipping build, using specified image: $ExistingImage" -ForegroundColor Yellow
}

Write-Host "8. Writing temporary env.yaml to prevent comma-parsing errors..." -ForegroundColor Yellow
$envYaml = @"
QDRANT_URL: "$QdrantUrl"
ALLOWED_ORIGINS: "$AllowedOrigins"
WORKERS: "1"
LLM_MODEL: "$LlmModel"
LOG_LEVEL: "INFO"
ANNA_APP_SLUG: "anna-document-workspace"
ANNA_APP_VERSION: "1.0.0"
ANNA_APP_MODE: "cloud-compatible"
AWS_ENDPOINT_URL_S3: "$AwsEndpoint"
AWS_REGION: "$AwsRegion"
S3_BUCKET_NAME: "$S3Bucket"
"@

[System.IO.File]::WriteAllText("env.yaml", $envYaml, [System.Text.Encoding]::UTF8)

Write-Host "9. Deploying to Cloud Run with Cost & Concurrency Optimizations..." -ForegroundColor Yellow
try {
    gcloud run deploy "$ServiceName" `
      --image "$Image" `
      --region "$Region" `
      --platform managed `
      --allow-unauthenticated `
      --service-account "$ServiceAccount" `
      --memory 1536Mi `
      --cpu 1 `
      --concurrency 80 `
      --min-instances 0 `
      --max-instances 5 `
      --timeout 300 `
      --cpu-throttling `
      --env-vars-file="env.yaml" `
      --set-secrets "GROQ_API_KEY=groq-api-key:latest,QDRANT_API_KEY=qdrant-api-key:latest,AWS_ACCESS_KEY_ID=supabase-s3-access-key:latest,AWS_SECRET_ACCESS_KEY=supabase-s3-secret-key:latest" `
      --quiet
} finally {
    if (Test-Path "env.yaml") {
        Remove-Item "env.yaml" -Force -ErrorAction SilentlyContinue
    }
}

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Green
Write-Host " Deployment SUCCESSFUL!" -ForegroundColor Green
Write-Host " Service URL:" -ForegroundColor Green
$ServiceUrl = gcloud run services describe "$ServiceName" --region "$Region" --project="$ProjectId" --format='value(status.url)'
Write-Host $ServiceUrl -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Green
