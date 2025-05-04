#!/bin/bash

# Create directory on cluster if it doesn't exist
ssh lr3956@della.princeton.edu "mkdir -p /scratch/network/lr3956/thesis_project"

# Transfer all files
scp -r ./* lr3956@della.princeton.edu:/scratch/network/lr3956/thesis_project/

echo "Files transferred successfully!"
echo "Now you can:"
echo "1. SSH into the cluster: ssh lr3956@della.princeton.edu"
echo "2. Navigate to your project: cd /scratch/network/lr3956/thesis_project"
echo "3. Make the Slurm script executable: chmod +x run_thesis.slurm"
echo "4. Submit the job: sbatch run_thesis.slurm" 