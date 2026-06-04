#!/bin/bash

# Exit on error
set -e

echo "Starting RFM Analysis and Deep Learning Pipeline..."

# Ensure we are in the project directory
cd "$(dirname "$0")"

echo "Running Standard RFM Clustering and Plots..."
python3 plot_frequency_elbow.py
python3 plot_3d_cleaning.py
python3 plot_clusters_3d.py
python3 plot_segment_distribution.py
python3 plot_rfm_boxplots.py

echo "Running Hypothesis Testing..."
python3 hypothesis_testing.py

echo "Running Deep Learning Model Training and Evaluation..."
echo "Generating Deep Learning Plots..."
python3 plot_model_metrics.py

echo "Generating Outlier Deep Dive & Cluster Comparison..."
python3 plot_outlier_comparison.py

echo "Generating Table 4 Statistics (Cluster Means)..."
python3 generate_cluster_stats.py

echo "Analysis Complete. Check output files in this directory."
