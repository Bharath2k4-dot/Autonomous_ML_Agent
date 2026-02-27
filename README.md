# Autonomous ML Lifecycle Agent (Mini Michelangelo Inspired) — Version 1

## What this project does
This project is a Python-based autonomous ML pipeline that:
1. Loads a CSV dataset
2. Identifies the target column (auto or user-provided)
3. Detects whether the task is Classification or Regression
4. Builds preprocessing for numeric + categorical columns
5. Trains multiple candidate models
6. Evaluates them using correct metrics
7. Automatically selects the best model
8. Saves the full preprocessing + model pipeline for deployment
9. Generates run reports (JSON + TXT)

## Why this matters (real-world)
This simulates an enterprise ML platform workflow (like Uber Michelangelo):
- repeatable model training runs
- standardized evaluation
- automated model selection
- deployable artifacts

## Project structure
- main.py                     -> runs the agent
- data/                       -> your CSV datasets
- models/                     -> saved best model pipeline
- reports/                    -> run reports (metrics + summary)

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt