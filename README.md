# ML API

A machine learning SVC model served as a live REST API.

## Live Demo
[https://ml-api-xxxx.onrender.com/docs](https://churn-api-l8su.onrender.com/docs)

## Tech Stack
- Python, Scikit-learn
- FastAPI
- Docker
- Deployed on Render

## Run locally
docker build -t ml-api .
docker run -p 8000:8000 ml-api
