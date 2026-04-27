# CS 643 PA2 - Wine Quality Prediction

Apache Spark + MLlib wine quality prediction with parallel training and Docker deployment.

## Docker Hub
https://hub.docker.com/r/bhavanavuttunoori/wine-predictor

## Run with Docker
docker pull bhavanavuttunoori/wine-predictor:1.0
docker run --rm -v /path/to/data:/app/data bhavanavuttunoori/wine-predictor:1.0 /app/data/TestDataset.csv /app/model
