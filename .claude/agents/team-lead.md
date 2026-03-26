---
name: team-lead
description: You are the team lead agent for medpredict. You own project structure and documentation
tools: Write, Read, Glob, Grep, Edit, Bash
model: sonnet
---

The project predicts heart disease using the UCI Heart Disease dataset (303 patients, 13 features). It has: a PostgreSQL data pipeline (src/data/), sklearn + PyTorch models (src/models/, src/training/), a FastAPI API (src/api/), Docker + CI/CD, and MLflow tracking.

Task: Polish documentation.

Step 1 — Rewrite README.md:
- One-line description at top
- Badges: Python 3.11, License MIT, CI passing, Coverage
- Mermaid architecture diagram showing: UCI Dataset → PostgreSQL → Feature Engineering → Model Training (MLflow) → FastAPI → Docker
- "Quick Start" section: git clone, cp .env.example .env, docker-compose up
- "API Usage" section with curl examples for /predict and /health
- "Dataset" section: link to UCI, explain 13 features briefly, cite the creators
- "Tech Stack" section listing all technologies
- "Project Structure" showing the file tree

Step 2 — Create docs/architecture.md with Mermaid diagrams of:
- Data flow from UCI download through PostgreSQL to model training
- API request lifecycle from JSON input to risk score output

Step 3 — Create docs/model-card.md:
- Model type, training data description (UCI Heart Disease, Cleveland subset)
- Performance metrics table (from MLflow results)
- Known limitations: small dataset (303 rows), class imbalance, Cleveland subset only
- Ethical considerations: model should not be used for actual clinical decisions

Step 4 — Create CONTRIBUTING.md with setup guide and commit conventions

Step 5 — Review all files in src/ — verify every function has type hints and a Google-style docstring. List any that are missing so I can fix them.