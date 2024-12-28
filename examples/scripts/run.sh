#!/bin/bash

# Check if model file is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <model_name> [node_list]"
    echo "Example: $0 cnn \"2 4 8 16\""
    echo "If node_list is not provided, defaults to \"2 4 8 16\""
    exit 1
fi

MODEL_NAME="$1"
MODEL_FILE="../sequential/${MODEL_NAME}.py"
NODE_LIST=${2:-"2 4 8 16"}  # Use provided node list or default to "2 4 8 16"

# Check if the model file exists
if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: Model file $MODEL_FILE does not exist"
    exit 1
fi

# Directory for storing results
RESULTS_DIR="benchmark_results/${MODEL_NAME}"
mkdir -p "$RESULTS_DIR"

# Log file for compilation of results
LOG_FILE="$RESULTS_DIR/benchmark_summary.txt"
echo "DiLoCo Benchmark Results for ${MODEL_NAME}" > "$LOG_FILE"
echo "=================================" >> "$LOG_FILE"
echo "Date: $(date)" >> "$LOG_FILE"
echo "Model: ${MODEL_NAME}" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Function to run experiment and capture metrics
run_experiment() {
    local num_nodes=$1
    local output_file="$RESULTS_DIR/run_n${num_nodes}.txt"
    local temp_script="$RESULTS_DIR/temp_${MODEL_NAME}_n${num_nodes}.py"
    
    echo "Running experiment with ${num_nodes} nodes for ${MODEL_NAME}..."
    echo "Results will be saved to ${output_file}"
    
    # Record start time
    start_time=$(date +%s)
    
    # Create temporary script with modified node count
    cp "$MODEL_FILE" "$temp_script"
    sed -i "s/num_nodes=[0-9][0-9]*/num_nodes=${num_nodes}/" "$temp_script"
    
    # Run the training script and capture output
    CUDA_VISIBLE_DEVICES=0,1,2,3 python "$temp_script" 2>&1 | tee "$output_file"
    
    # Record end time and calculate duration
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    # Extract metrics from the output file
    final_loss=$(grep "Final loss:" "$output_file" | tail -n 1 | awk '{print $NF}' || echo "N/A")
    
    # Extract GPU memory usage if available
    gpu_mem=$(grep "GPU memory:" "$output_file" | tail -n 1 | awk '{print $NF}' || echo "N/A")
    
    # Log results
    {
        echo "Nodes: $num_nodes"
        echo "Duration: $duration seconds"
        echo "Final Loss: $final_loss"
        echo "GPU Memory: $gpu_mem"
        echo "------------------------"
    } >> "$LOG_FILE"
    
    # Cleanup
    rm "$temp_script"
}

# Run experiments for specified node counts
for nodes in $NODE_LIST; do
    run_experiment "$nodes"
done

# Generate summary plots using Python
python -c '
import matplotlib.pyplot as plt
import re
import numpy as np

# Parse results
nodes = []
times = []
losses = []

try:
    with open("'"$LOG_FILE"'", "r") as f:
        content = f.read()
        node_matches = re.findall(r"Nodes: (\d+)", content)
        time_matches = re.findall(r"Duration: (\d+)", content)
        loss_matches = re.findall(r"Final Loss: ([\d.]+)", content)
        
        nodes = [int(n) for n in node_matches]
        times = [int(t) for t in time_matches]
        losses = [float(l) for l in loss_matches if l != "N/A"]
    
    # Create figure with subplots
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Wall-clock time vs nodes
    plt.subplot(1, 3, 1)
    plt.plot(nodes, times, "bo-")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Wall-clock Time (s)")
    plt.title("'"$MODEL_NAME"': Scaling Performance")
    plt.grid(True)
    
    # Plot 2: Efficiency
    baseline_time = times[0]  # time with minimum nodes
    ideal_times = [baseline_time * nodes[0] / n for n in nodes]
    efficiency = [100 * i / r for i, r in zip(ideal_times, times)]
    
    plt.subplot(1, 3, 2)
    plt.plot(nodes, efficiency, "ro-")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Efficiency (%)")
    plt.title("'"$MODEL_NAME"': Scaling Efficiency")
    plt.grid(True)
    
    # Plot 3: Loss vs Nodes (only if valid losses exist)
    if losses:
        plt.subplot(1, 3, 3)
        plt.plot(nodes[:len(losses)], losses, "go-")
        plt.xlabel("Number of Nodes")
        plt.ylabel("Final Loss")
        plt.title("'"$MODEL_NAME"': Final Loss")
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("'"$RESULTS_DIR"'/scaling_results.png")
except Exception as e:
    print(f"Error generating plots: {e}")
'

echo "Benchmark complete for ${MODEL_NAME}!"
echo "Results saved to $RESULTS_DIR"
echo "See $LOG_FILE for detailed results"
echo "See $RESULTS_DIR/scaling_results.png for scaling plots"