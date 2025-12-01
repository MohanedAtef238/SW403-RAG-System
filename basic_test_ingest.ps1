# Basic test ingestion script for SW403 RAG system
# Ingests project1 and project2 files into both P1 and P2 prototypes

Write-Host "SW403 Basic Test Ingestion"
Write-Host "============================="
Write-Host ""

# Check if services are running
Write-Host "Checking services..."
try {
    $p1Health = Invoke-WebRequest -Uri "http://localhost:8001/health"
    Write-Host "✓ P1 service is healthy"
} catch {
    Write-Host "✗ P1 service is not accessible"
    Write-Host "  Start services with: docker compose up -d"
    exit 1
}

try {
    $p2Health = Invoke-WebRequest -Uri "http://localhost:8002/health"
    Write-Host "✓ P2 service is healthy"
} catch {
    Write-Host "✗ P2 service is not accessible"
    Write-Host "  Start services with: docker compose up -d"
    exit 1
}

Write-Host ""
Write-Host "Starting ingestion..."
Write-Host ""

# Ingest into P1
Write-Host "[1/4] Ingesting project1 files into P1..."
$bodyP1Project1 = @{
    file_paths = @(
        "/app/shared_data/project1/AStarAgent.py",
        "/app/shared_data/project1/BFSAgent.py",
        "/app/shared_data/project1/DFSAgent.py",
        "/app/shared_data/project1/maze.py",
        "/app/shared_data/project1/PapaAgent.py",
        "/app/shared_data/project1/SmartAlgo.py",
        "/app/shared_data/project1/QBot.py"
    )
    recreate_collection = $true
} | ConvertTo-Json

try {
    $response = Invoke-WebRequest -Uri "http://localhost:8001/ingest" `
        -Method POST `
        -ContentType "application/json" `
        -Body $bodyP1Project1 `
        -ErrorAction Stop
    
    $result = $response.Content | ConvertFrom-Json
    Write-Host "  ✓ Processed $($result.functions_processed) functions"
    Write-Host "  Time: $($result.processing_time.ToString('F2'))s"
} catch {
    Write-Host "  ✗ Failed: $_"
    exit 1
}

Write-Host ""

# Ingest project2 into P1
Write-Host "[2/4] Ingesting project2 files into P1..."
$bodyP1Project2 = @{
    file_paths = @(
        "/app/shared_data/project2/processing/parsers.py",
        "/app/shared_data/project2/processing/preprocess.py"
    )
    recreate_collection = $false
} | ConvertTo-Json

try {
    $response = Invoke-WebRequest -Uri "http://localhost:8001/ingest" `
        -Method POST `
        -ContentType "application/json" `
        -Body $bodyP1Project2 `
        -ErrorAction Stop
    
    $result = $response.Content | ConvertFrom-Json
    Write-Host "  ✓ Processed $($result.functions_processed) functions"
    Write-Host "  Time: $($result.processing_time.ToString('F2'))s"
} catch {
    Write-Host "  ✗ Failed: $_"
    exit 1
}

Write-Host ""

# Ingest into P2
Write-Host "[3/4] Ingesting project1 files into P2..."
$bodyP2Project1 = @{
    file_paths = @(
        "/app/shared_data/project1/AStarAgent.py",
        "/app/shared_data/project1/BFSAgent.py",
        "/app/shared_data/project1/DFSAgent.py",
        "/app/shared_data/project1/maze.py",
        "/app/shared_data/project1/PapaAgent.py",
        "/app/shared_data/project1/SmartAlgo.py",
        "/app/shared_data/project1/QBot.py"
    )
    recreate_collection = $true
} | ConvertTo-Json

try {
    $response = Invoke-WebRequest -Uri "http://localhost:8002/ingest" `
        -Method POST `
        -ContentType "application/json" `
        -Body $bodyP2Project1 `
        -ErrorAction Stop
    
    $result = $response.Content | ConvertFrom-Json
    Write-Host "  ✓ Processed $($result.functions_processed) functions"
    Write-Host "  Time: $($result.processing_time.ToString('F2'))s"
} catch {
    Write-Host "  ✗ Failed: $_"
    exit 1
}

Write-Host ""

# Ingest project2 into P2
Write-Host "[4/4] Ingesting project2 files into P2..."
$bodyP2Project2 = @{
    file_paths = @(
        "/app/shared_data/project2/processing/parsers.py",
        "/app/shared_data/project2/processing/preprocess.py"
    )
    recreate_collection = $false
} | ConvertTo-Json

try {
    $response = Invoke-WebRequest -Uri "http://localhost:8002/ingest" `
        -Method POST `
        -ContentType "application/json" `
        -Body $bodyP2Project2 `
        -ErrorAction Stop
    
    $result = $response.Content | ConvertFrom-Json
    Write-Host "  ✓ Processed $($result.functions_processed) functions"
    Write-Host "  Time: $($result.processing_time.ToString('F2'))s"
} catch {
    Write-Host "  ✗ Failed: $_"
    exit 1
}

Write-Host ""
Write-Host "Ingestion completed successfully!"
Write-Host ""

# Show collection info
Write-Host "Collection Summary:"
Write-Host "==================="

try {
    $p1Info = Invoke-WebRequest -Uri "http://localhost:8001/collection/info" | ConvertFrom-Json
    Write-Host "P1 (functions_p1): $($p1Info.points_count) functions"
} catch {
    Write-Host "P1: Unable to fetch info"
}

try {
    $p2Info = Invoke-WebRequest -Uri "http://localhost:8002/collection/info" | ConvertFrom-Json
    Write-Host "P2 (functions_p2): $($p2Info.points_count) functions"
} catch {
    Write-Host "P2: Unable to fetch info"
}

Write-Host ""
Write-Host "Ready to run evaluation:"
Write-Host "  python -m evaluation.runner --ground-truth shared_data/ground_truth.json --output-dir results --top-k 5"
