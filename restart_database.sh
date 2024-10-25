#!/bin/bash

# Stop the Docker container
sudo docker stop querylake_db

# Remove the Docker container
sudo docker rm querylake_db


# Pull the latest Docker image for ParadeDB
sudo docker pull paradedb/paradedb:latest

# Remove the db_data directory
# The script uses "$(dirname "$0")" to get the directory where the script is located
# sudo rm -rf "$(dirname "$0")/db_data"
sudo docker volume prune -y

# Free disk space by removing unused Docker volumes
sudo docker volume prune

# Start the Docker container
# sudo docker compose -f "$(dirname "$0")/docker-compose-only-db.yml" up -d

sudo docker run \
  --name querylake_db \
  -e POSTGRES_USER="querylake_access" \
  -e POSTGRES_PASSWORD="querylake_access_password" \
  -e POSTGRES_DB="querylake_database" \
  -v querylake_database_volume:/var/lib/postgresql/data/ \
  -p 5444:5432 \
  -d \
  paradedb/paradedb:latest