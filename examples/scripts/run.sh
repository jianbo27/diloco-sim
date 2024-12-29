#!/bin/bash

# Check if model file is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <model_name> [node_list]"
    echo "Example: $0 cnn \"1 2 4 8 16\""
    echo "If node_list is not provided, defaults to \"1 2 4 8 16\""
    exit 1
fi

MODEL_NAME="$1"
MODEL_FILE="../sequential/${MODEL_NAME}.py"
NODE_LIST=${2:-"1 2 4 8 16"}

if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: Model file $MODEL_FILE does not exist"
    exit 1
fi

RESULTS_DIR="benchmark_results/${MODEL_NAME}"
mkdir -p "$RESULTS_DIR"

# Log file for results
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
    local loss_file="$RESULTS_DIR/losses_n${num_nodes}.csv"
    
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
    
    # Process output into clean CSV format
    echo "step,loss,accuracy" > "$loss_file"
    grep -A 1 "Eval Loss:" "$output_file" | awk '
        BEGIN {step=0}
        /Eval Loss:/ {
            loss=$3
            getline
            acc=$3
            printf "%d,%.4f,%.4f\n", step, loss, acc
            step+=500
        }
    ' >> "$loss_file"
    
    # Extract GPU memory usage
    gpu_mem=$(grep "GPU memory:" "$output_file" | tail -n 1 | awk '{print $(NF-1)}' || echo "N/A")
    
    # Log results
    {
        echo "Nodes: $num_nodes"
        echo "Duration: $duration seconds"
        echo "GPU Memory: $gpu_mem GB"
        echo "Loss history saved to: $loss_file"
        echo "------------------------"
    } >> "$LOG_FILE"
    
    # Cleanup
    rm "$temp_script"
}

# Run experiments for specified node counts
for nodes in $NODE_LIST; do
    run_experiment "$nodes"
done

# Modified plotting script
python -c '
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import os

try:
    results_dir = "'"$RESULTS_DIR"'"
    node_counts = [int(n) for n in "'"$NODE_LIST"'".split()]
    
    # Read timing data
    with open("'"$LOG_FILE"'", "r") as f:
        lines = f.readlines()
        times = []
        nodes = []
        for line in lines:
            if line.startswith("Duration:"):
                times.append(int(line.split()[1]))
            elif line.startswith("Nodes:"):
                nodes.append(int(line.split()[1]))
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Loss curves for all node counts
    ax1 = fig.add_subplot(221)
    max_steps = 0
    
    # Color map for different node counts
    colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, len(node_counts)))
    
    for idx, n in enumerate(node_counts):
        loss_file = os.path.join(results_dir, f"losses_n{n}.csv")
        if os.path.exists(loss_file):
            df = pd.read_csv(loss_file)
            max_steps = max(max_steps, df["step"].max())
            ax1.plot(df["step"], df["loss"], 
                    label=f"{n} nodes",
                    color=colors[idx],
                    marker="o",
                    markersize=4,
                    markevery=2)
    
    ax1.set_xlabel("Steps (×500)")
    ax1.set_ylabel("Loss")
    ax1.set_title("Eval Loss vs Steps")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Create evenly spaced tick positions
    tick_positions = np.linspace(0, max_steps, 10)
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels([f"{int(x/500)}" for x in tick_positions])
    
    # Plot 2: Accuracy curves for all node counts
    ax2 = fig.add_subplot(222)
    for idx, n in enumerate(node_counts):
        loss_file = os.path.join(results_dir, f"losses_n{n}.csv")
        if os.path.exists(loss_file):
            df = pd.read_csv(loss_file)
            ax2.plot(df["step"], df["accuracy"],
                    label=f"{n} nodes",
                    color=colors[idx],
                    marker="o",
                    markersize=4,
                    markevery=2)
    
    ax2.set_xlabel("Steps (×500)")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Eval Accuracy vs Steps")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels([f"{int(x/500)}" for x in tick_positions])
    
    # Plot 3: Wall-clock time vs nodes
    ax3 = fig.add_subplot(223)
    ax3.plot(nodes, times, "bo-", linewidth=2)
    ax3.set_xlabel("Number of Nodes")
    ax3.set_ylabel("Wall-clock Time (s)")
    ax3.set_title("Scaling Performance")
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Efficiency analysis
    ax4 = fig.add_subplot(224)
    baseline_nodes = nodes[0]
    baseline_time = times[0]
    ideal_times = np.array([baseline_time * baseline_nodes / n for n in nodes])
    efficiency = 100 * ideal_times / np.array(times)
    
    ax4.plot(nodes, efficiency, "ro-", linewidth=2)
    ax4.set_xlabel("Number of Nodes")
    ax4.set_ylabel("Parallel Efficiency (%)")
    ax4.set_title("Scaling Efficiency")
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "benchmark_results.png"), dpi=300)
    
except Exception as e:
    import traceback
    print(f"Error generating plots: {e}")
    print(traceback.format_exc())
'

echo "Benchmark complete for ${MODEL_NAME}!"
echo "Results saved to $RESULTS_DIR"
echo "See $LOG_FILE for detailed results"
echo "See $RESULTS_DIR/benchmark_results.png for plots"