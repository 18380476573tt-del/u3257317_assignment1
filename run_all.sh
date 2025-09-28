#!/bin/bash
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setting up Git LFS..."
git lfs install

echo "Running Part A..."
python Assignment1_PartA.py

echo "Running Part B..."
python Assignment1_PartB.py

echo "All tasks completed!"
