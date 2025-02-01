# SmolLM Deployment using FastAPI and Docker
This repository contains the code for deploying a SmolLM model using FastAPI and Docker. 
There are two parts to this deployment:
1. The FastAPI backend which hosts the model and serves predictions.
2. The FastAPI frontend which serves the UI for the model where users can input text and get predictions.

## Steps to deploy the model
1. Clone the repository
2. Run `docker-compose up --build`
3. Navigate to `http://localhost:8000` to access the UI
