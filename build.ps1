# Build script for SW403 RAG System
# Builds base image once, then P1 and P2 inherit from it

Write-Host "Building SW403 RAG System..."

# Step 1: Check if base image exists
$baseExists = docker images -q sw403-base
if (-not $baseExists) {
    Write-Host ""
    Write-Host "[1/3] Building base image (sw403-base) with shared dependencies..."
    Write-Host "This will take a few minutes (only needed once)..."
    docker build -t sw403-base -f Dockerfile .
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to build base image!"
        exit 1
    }
    
    Write-Host "Base image built successfully"
} else {
    Write-Host ""
    Write-Host "[1/3] Base image (sw403-base) already exists (skipping)"
    Write-Host "To rebuild base image (if dependencies changed): docker build -t sw403-base ."
}

# Step 2: Build P1 and P2 (they inherit from sw403-base)
Write-Host ""
Write-Host "[2/3] Building P1 and P2 containers..."
docker compose build p1 p2

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to build P1/P2 containers!"
    exit 1
}

Write-Host "P1 and P2 built successfully"

# Step 3: Start all services
Write-Host ""
Write-Host "[3/3] Starting all services..."
docker compose up -d qdrant-p1 qdrant-p2 p1 p2

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to start services!"
    exit 1
}

Write-Host ""
Write-Host "All services started successfully!"
Write-Host ""
Write-Host "Services:"
Write-Host "  - Qdrant (P1): http://localhost:6333"
Write-Host "  - Qdrant (P2): http://localhost:7333"
Write-Host "  - P1 API:  http://localhost:8001"
Write-Host "  - P2 API:  http://localhost:8002"
Write-Host ""
Write-Host "Check status: docker compose ps"
Write-Host "View logs:    docker compose logs -f"
