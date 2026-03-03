# ClearML + GitHub Actions (Simple End-to-End Demo)

This repo demonstrates a small, easy-to-understand end-to-end flow:
- ClearML tracking (Tasks, metrics, params)
- Artifacts (model + plots + processed data)
- Datasets (if supported by your ClearML server; otherwise fallback)
- Pipeline (PipelineDecorator)
- Model Registry (publishes the trained model)
- Runs inside GitHub Actions

## 1) Prerequisites
- A working ClearML Server (self-hosted)
- You can reach it from GitHub Actions (public IP or VPN/runner in same network)
- Create a ClearML user and get API keys

## 2) Configure GitHub Secrets
In your GitHub repo: Settings → Secrets and variables → Actions → New repository secret

Add:
- CLEARML_API_ACCESS_KEY
- CLEARML_API_SECRET_KEY
- CLEARML_API_HOST      (example: http://YOUR_SERVER:8008)
- CLEARML_WEB_HOST      (example: http://YOUR_SERVER:8080)
- CLEARML_FILES_HOST    (example: http://YOUR_SERVER:8081)

## 3) Run
### Local
```bash
pip install -r requirements.txt
python -m src.pipeline --project "ClearML_GHA_Demo" --name "BreastCancer_Pipeline"

