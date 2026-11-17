# Update Secret Manager Secrets and Cloud Run without rebuilding Docker Image
$ErrorActionPreference = "Continue"

$ProjectId = "banded-advice-437316-g8"
$Region = "us-central1"
$ServiceName = "anna-document-workspace"
$ServiceAccount = "anna-document-runtime@$ProjectId.iam.gserviceaccount.com"

# Load .env
if (-not (Test-Path ".env")) {
    throw ".env file not found!"
}

Write-Host "1. Reading secrets from .env..." -ForegroundColor Cyan
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

$secrets = @{
    "groq-api-key" = $env:GROQ_API_KEY
    "qdrant-api-key" = $env:QDRANT_API_KEY
    "supabase-s3-access-key" = $env:AWS_ACCESS_KEY_ID
    "supabase-s3-secret-key" = $env:AWS_SECRET_ACCESS_KEY
}

foreach ($key in $secrets.Keys) {
    if (-not $secrets[$key]) {
        throw "Missing required value for $key in .env"
    }
}

Write-Host "2. Creating / updating Secret Manager secrets in GCP..." -ForegroundColor Cyan
foreach ($name in $secrets.Keys) {
    $val = $secrets[$name].Trim()
    $tempFile = [System.IO.Path]::GetTempFileName()
    try {
        [System.IO.File]::WriteAllText($tempFile, $val, (New-Object System.Text.UTF8Encoding($false)))
        $describe = gcloud secrets describe $name --project="$ProjectId" 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Creating secret: $name" -ForegroundColor Yellow
            gcloud secrets create $name --project="$ProjectId" --replication-policy="automatic" --data-file="$tempFile" --quiet
        } else {
            Write-Host "Adding new version for secret: $name" -ForegroundColor Yellow
            gcloud secrets versions add $name --project="$ProjectId" --data-file="$tempFile" --quiet
        }
    } finally {
        if (Test-Path $tempFile) {
            Remove-Item $tempFile -Force -ErrorAction SilentlyContinue
        }
    }
}

Write-Host "3. Granting secretAccessor role to service account..." -ForegroundColor Cyan
foreach ($name in $secrets.Keys) {
    gcloud secrets add-iam-policy-binding $name `
      --project="$ProjectId" `
      --member="serviceAccount:$ServiceAccount" `
      --role="roles/secretmanager.secretAccessor" `
      --quiet *>$null
}

Write-Host "4. Updating Cloud Run service to remount secrets (re-uses existing image)..." -ForegroundColor Cyan
gcloud run services update "$ServiceName" `
  --region="$Region" `
  --project="$ProjectId" `
  --set-secrets="GROQ_API_KEY=groq-api-key:latest,QDRANT_API_KEY=qdrant-api-key:latest,AWS_ACCESS_KEY_ID=supabase-s3-access-key:latest,AWS_SECRET_ACCESS_KEY=supabase-s3-secret-key:latest" `
  --quiet

Write-Host "5. Verifying deployment health..." -ForegroundColor Green
$ServiceUrl = (gcloud run services describe "$ServiceName" --region="$Region" --project="$ProjectId" --format='value(status.url)').Trim()
Write-Host "Service URL: $ServiceUrl" -ForegroundColor Cyan
