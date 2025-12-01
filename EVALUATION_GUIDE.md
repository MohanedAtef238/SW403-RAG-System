# SW403 RAG Evaluation Guide

This guide walks through executing baseline experiments and generating the experimental results report.

## Prerequisites

1. **Dependencies installed:**
   ```bash
   uv sync
   ```

2. **Docker services running:**
   ```bash
   docker compose up --build
   ```
   - Qdrant: port 6333
   - P1 API: port 8001
   - P2 API: port 8002

## Step-by-Step Execution

### Step 1: Ingest Data into Both Prototypes


Ingest all Python files from project1 and project2 into both P1 and P2 collections using PowerShell commands.

**Ingest into P1 (port 8001):**
```powershell
# Project 1 files
Invoke-WebRequest -Uri "http://localhost:8001/ingest" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{
    "file_paths": [
      "/app/shared_data/project1/AStarAgent.py",
      "/app/shared_data/project1/BFSAgent.py",
      "/app/shared_data/project1/DFSAgent.py",
      "/app/shared_data/project1/GBFSAgent.py",
      "/app/shared_data/project1/UCSAgent.py",
      "/app/shared_data/project1/IDSAgent.py",
      "/app/shared_data/project1/HillClimbing.py",
      "/app/shared_data/project1/SimulatedAnnealing.py",
      "/app/shared_data/project1/Genetic_algorithmAgent.py",
      "/app/shared_data/project1/PapaAgent.py",
      "/app/shared_data/project1/SmartAlgo.py",
      "/app/shared_data/project1/QBot.py",
      "/app/shared_data/project1/maze.py",
      "/app/shared_data/project1/tiles.py",
      "/app/shared_data/project1/colors.py",
      "/app/shared_data/project1/main.py"
    ],
    "recreate_collection": true
  }'

# Project 2 files
Invoke-WebRequest -Uri "http://localhost:8001/ingest" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{
    "file_paths": [
      "/app/shared_data/project2/processing/parsers.py",
      "/app/shared_data/project2/processing/preprocess.py"
    ],
    "recreate_collection": false
  }'
```


**Ingest into P2 (port 8002):**
```powershell
# Project 1 files
Invoke-WebRequest -Uri "http://localhost:8002/ingest" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{
    "file_paths": [
      "/app/shared_data/project1/AStarAgent.py",
      "/app/shared_data/project1/BFSAgent.py",
      "/app/shared_data/project1/DFSAgent.py",
      "/app/shared_data/project1/GBFSAgent.py",
      "/app/shared_data/project1/UCSAgent.py",
      "/app/shared_data/project1/IDSAgent.py",
      "/app/shared_data/project1/HillClimbing.py",
      "/app/shared_data/project1/SimulatedAnnealing.py",
      "/app/shared_data/project1/Genetic_algorithmAgent.py",
      "/app/shared_data/project1/PapaAgent.py",
      "/app/shared_data/project1/SmartAlgo.py",
      "/app/shared_data/project1/QBot.py",
      "/app/shared_data/project1/maze.py",
      "/app/shared_data/project1/tiles.py",
      "/app/shared_data/project1/colors.py",
      "/app/shared_data/project1/main.py"
    ],
    "recreate_collection": true
  }'

# Project 2 files
Invoke-WebRequest -Uri "http://localhost:8002/ingest" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{
    "file_paths": [
      "/app/shared_data/project2/processing/parsers.py",
      "/app/shared_data/project2/processing/preprocess.py"
    ],
    "recreate_collection": false
  }'
```


**Check ingestion status:**
```powershell
# Check health
Invoke-WebRequest -Uri "http://localhost:8001/health"
Invoke-WebRequest -Uri "http://localhost:8002/health"

# Check collection info (vector count, etc.)
Invoke-WebRequest -Uri "http://localhost:8001/collection/info"
Invoke-WebRequest -Uri "http://localhost:8002/collection/info"
```

### Step 2: Run Evaluation

Execute all 30 ground truth queries against both P1 and P2.

```bash
uv run python evaluation/runner.py \
  --ground-truth shared_data/ground_truth.json \
  --output-dir results \
  --top-k 5
```

**Expected output:**
- `results/p1_results.json` - P1 evaluation metrics
- `results/p2_results.json` - P2 evaluation metrics
- `results/evaluation_results.json` - Combined results

**Optional flags:**
- `--skip-p1` - Skip P1 evaluation
- `--skip-p2` - Skip P2 evaluation
- `--p1-url http://localhost:8001` - Custom P1 URL
- `--p2-url http://localhost:8002` - Custom P2 URL

### Step 3: Generate Error Analysis

Classify errors and create detailed error tables.

```bash
uv run python evaluation/analyzer.py \
  --results-dir results \
  --ground-truth shared_data/ground_truth.json
```

**Expected output:**
- `results/error_analysis.csv` - Error table in CSV format
- `results/error_analysis.md` - Error table in Markdown format
- `results/statistical_comparison.json` - T-test results with confidence intervals

### Step 4: Generate Visualizations

Create all charts and graphs for the experimental report.

```bash
uv run python evaluation/visualize.py --results-dir results
```

**Expected output in `results/figures/`:**
- `accuracy_by_category.png` - Bar chart of exact match rates
- `metrics_comparison.png` - IR metrics comparison (MRR, P@K, R@K, NDCG@K)
- `similarity_distributions.png` - Histogram of similarity scores
- `partial_credit_boxplot.png` - Boxplot by category
- `confidence_intervals.png` - Mean scores with 95% CI

## Quick Run (All Steps)

If services are already running and data is ingested:

```bash
# Run evaluation
uv run python evaluation/runner.py

# Analyze errors and compare
uv run python evaluation/analyzer.py

# Generate visualizations
uv run python evaluation/visualize.py
```

## Results Structure

```
results/
├── p1_results.json              # P1 evaluation metrics
├── p2_results.json              # P2 evaluation metrics
├── evaluation_results.json      # Combined results
├── error_analysis.csv           # Error classification table (CSV)
├── error_analysis.md            # Error classification table (Markdown)
├── statistical_comparison.json  # T-test results with CI
└── figures/                     # All visualizations
    ├── accuracy_by_category.png
    ├── metrics_comparison.png
    ├── similarity_distributions.png
    ├── partial_credit_boxplot.png
    └── confidence_intervals.png
```

## Troubleshooting

**Services not accessible:**
```bash
# Check if containers are running
docker ps

# Restart services
docker compose down
docker compose up --build
```

**Empty results:**
- Verify data was ingested: `Invoke-WebRequest -Uri "http://localhost:8001/collection/info"`
- Check collection size is > 0

**Import errors:**
- Run `uv sync` to install all dependencies
- Ensure you're in the project root directory

## Next Steps

After generating all results:
1. Review error analysis tables to understand failure patterns
2. Examine visualizations for P1 vs P2 comparison
3. Check statistical significance in `statistical_comparison.json`
4. Use results to create the experimental results report in `docs/experimental_results.md`
