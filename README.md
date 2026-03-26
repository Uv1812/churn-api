# Customer churn prediction API

A machine learning model served as a live REST API.

## Live Demo
https://churn-api-l8su.onrender.com/docs

## Tech Stack
- Python, Scikit-learn
- FastAPI
- Docker
- Deployed on Render

## Run locally
docker build -t churn-api .
docker run -p 8000:8000 churn-api
