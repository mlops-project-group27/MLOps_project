#!/bin/bash
set -e

echo " Starting Credit Card Fraud Detection System"


REGION=europe-west1
PROJECT_ID=mlops-group27
REPO=mlops-docker

DATASET_IMG=mlops-dataset
TRAIN_IMG=mlops-train
BACKEND_IMG=mlops-api
FRONTEND_IMG=mlops-front


NETWORK=mlops-net

# Create network if missing
docker network inspect $NETWORK >/dev/null 2>&1 || docker network create $NETWORK

pull_or_build () {
  IMAGE_REMOTE=$1
  IMAGE_LOCAL=$2
  DOCKERFILE=$3

  echo "👉 Trying to pull $IMAGE_REMOTE"
  if docker pull $IMAGE_REMOTE; then
    echo "✅ Pulled $IMAGE_REMOTE"
    docker tag $IMAGE_REMOTE $IMAGE_LOCAL
  else
    echo "⚠️ Pull failed, building locally..."
    docker build -t $IMAGE_LOCAL -f $DOCKERFILE .
  fi
}


# ---- IMAGES ----
pull_or_build \
  europe-west1-docker.pkg.dev/mlops-group27/mlops-docker/mlops-dataset:latest \
  $DATASET_IMG \
  docker/Dockerfile.dataset

pull_or_build \
  europe-west1-docker.pkg.dev/mlops-group27/mlops-docker/mlops-train:latest \
  $TRAIN_IMG \
  docker/Dockerfile.train

pull_or_build \
  europe-west1-docker.pkg.dev/mlops-group27/mlops-docker/mlops-api:latest \
  $BACKEND_IMG \
  docker/Dockerfile.api

pull_or_build \
  europe-west1-docker.pkg.dev/mlops-group27/mlops-docker/mlops-front:latest \
  $FRONTEND_IMG \
  docker/Dockerfile.front

# ---- DATASET ----
echo " Running dataset container"
docker run --rm \
  --network $NETWORK \
  -v $(pwd)/dataset:/app/dataset \
  $DATASET_IMG

# ---- TRAINING ----
echo " Training model"
docker run --rm \
  --network $NETWORK \
  -v $(pwd)/dataset:/app/dataset \
  -v $(pwd)/models:/app/models \
  $TRAIN_IMG

# ---- BACKEND ----
echo " Starting backend"
docker rm -f backend >/dev/null 2>&1 || true
docker run -d \
  --name backend \
  --network $NETWORK \
  -p 8080:8080 \
  -v $(pwd)/models:/app/models \
  $BACKEND_IMG

# ---- FRONTEND ----
echo " Starting frontend"
docker rm -f frontend >/dev/null 2>&1 || true
docker run -d \
  --name frontend \
  --network $NETWORK \
  -p 8501:8501 \
  -e BACKEND_URL=http://backend:8080 \
  $FRONTEND_IMG

echo
echo "===================================="
echo " System is up and running"
echo "Backend:  http://localhost:8080"
echo "Frontend: http://localhost:8501"
echo "===================================="
