#!/bin/bash

# Stop the Docker container
sudo docker stop querylakebackend_postgres_db_1

# Remove the Docker container
sudo docker rm querylakebackend_postgres_db_1

# Remove the db_data directory
# The script uses "$(dirname "$0")" to get the directory where the script is located
sudo rm -rf "$(dirname "$0")/db_data"

# Start the Docker container
sudo docker-compose -f "$(dirname "$0")/docker-compose-only-db.yml" up -d