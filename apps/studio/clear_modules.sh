#!/bin/bash

# Get the directory of the current script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to that directory
cd "$DIR"

# Remove nodes
sudo rm -rf -f node_modules
sudo rm -rf -f .next
sudo rm -f bun.lockb