#!/bin/bash

# Check for required argument
if [ -z "$1" ]; then
  echo "Usage: $0 <download_directory>"
  exit 1
fi

# Set the target directory from the first argument
TARGET_DIR="$1"

# Create the directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Navigate into it
cd "$TARGET_DIR" || exit 1

# Download, unzip, and clean up
wget https://files.grouplens.org/datasets/movielens/ml-20m.zip
unzip ml-20m.zip
rm ml-20m.zip
