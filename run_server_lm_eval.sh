#!/bin/bash

# run_server_lm_eval.sh
# Script to loop through multiple models, start vLLM servers, and run evaluations

set -e  # Exit on any error
# Add error trapping
set -E  # Enable ERR trap inheritance
trap 'echo "ERROR: Script failed at line $LINENO with command: $BASH_COMMAND" >&2; handle_interrupt' ERR

#=============================================================================
# CONFIGURATION SECTION - Edit this to customize your evaluation
#=============================================================================

# Models to evaluate - Add or remove models as needed
MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8"
    "DESUCLUB/Llama-3.1-8B-Instruct-bf16-quantized.w8a8"
    # Add more models here:
    # "facebook/opt-125m"
    # "microsoft/DialoGPT-medium"
)

# Evaluation task configuration
TASK_NAME="hellaswag"          # Task to evaluate (hellaswag, openllm, gsm8k, etc.)
NUM_FEWSHOT=0                  # Number of few-shot examples
BATCH_SIZE=16                  # Batch size for evaluation
LIMIT=10                       # Limit samples for testing (empty = no limit, or set to number like 100)
LOG_SAMPLES=true               # Whether to log samples (true/false)

# Server configuration
VLLM_PORT=8000                 # Port for vLLM server
VLLM_HOST="0.0.0.0"           # Host for vLLM server
GPU_MEMORY_UTILIZATION=0.8     # GPU memory utilization (0.1 to 1.0)
MAX_MODEL_LEN=4096            # Maximum model context length
TENSOR_PARALLEL_SIZE=1         # Tensor parallel size
DTYPE="auto"                   # Data type (auto, float16, bfloat16, etc.)
ENABLE_SLEEP_MODE=true         # Enable sleep mode for memory management

# API configuration  
NUM_CONCURRENT=1               # Number of concurrent requests
MAX_RETRIES=3                  # Number of retry attempts
TOKENIZED_REQUESTS=false       # Whether to use tokenized requests

# Output configuration
OUTPUT_DIR="server_eval_output" # Directory to save results
EVAL_SCRIPT="server_eval.py"   # Evaluation script path

# Timeout configuration
SERVER_READY_TIMEOUT=1000      # Timeout in seconds to wait for server readiness
SERVER_SHUTDOWN_TIMEOUT=30     # Timeout in seconds for graceful server shutdown

#=============================================================================
# END CONFIGURATION SECTION
#=============================================================================

# Derived configuration (don't modify)
BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1/completions"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if server is ready
check_server_ready() {
    local max_attempts=$((SERVER_READY_TIMEOUT / 2))  # Check every 2 seconds
    local attempt=1
    
    log_info "Waiting for vLLM server to be ready (timeout: ${SERVER_READY_TIMEOUT}s)..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "http://${VLLM_HOST}:${VLLM_PORT}/health" > /dev/null 2>&1; then
            log_success "Server is ready after $((attempt * 2)) seconds!"
            return 0
        fi
        
        # Show progress every 10 attempts (20 seconds)
        if [ $((attempt % 10)) -eq 0 ]; then
            log_info "Still waiting... (${attempt}/${max_attempts} attempts, $((attempt * 2))s elapsed)"
        else
            echo -n "."
        fi
        
        sleep 2
        ((attempt++))
    done
    
    echo ""  # New line after dots
    log_error "Server failed to start after ${SERVER_READY_TIMEOUT} seconds"
    return 1
}

# Function to start vLLM server
start_vllm_server() {
    local model_name="$1"
    
    log_info "Starting vLLM server for model: $model_name"
    
    # Build vLLM command
    local vllm_cmd="vllm serve \"$model_name\""
    vllm_cmd+=" --host $VLLM_HOST"
    vllm_cmd+=" --port $VLLM_PORT"
    vllm_cmd+=" --gpu-memory-utilization $GPU_MEMORY_UTILIZATION"
    vllm_cmd+=" --dtype $DTYPE"
    vllm_cmd+=" --tensor-parallel-size $TENSOR_PARALLEL_SIZE"
    
    if [ -n "$MAX_MODEL_LEN" ]; then
        vllm_cmd+=" --max-model-len $MAX_MODEL_LEN"
    fi
    
    # Add sleep mode if enabled
    if [ "$ENABLE_SLEEP_MODE" = true ]; then
        vllm_cmd+=" --enable-sleep-mode"
    fi
    
    log_info "Command: $vllm_cmd"
    
    # Create log file name (replace / with __ for filesystem compatibility)
    local log_file="vllm_server_${model_name//\//__}.log"
    
    # Start server in background
    eval "$vllm_cmd" > "$log_file" 2>&1 &
    
    # Store PID
    VLLM_PID=$!
    echo $VLLM_PID > vllm_server.pid
    
    log_info "vLLM server started with PID: $VLLM_PID (log: $log_file)"
    
    # Wait for server to be ready
    if check_server_ready; then
        return 0
    else
        log_error "Server startup failed. Check log file: $log_file"
        stop_vllm_server
        return 1
    fi
}

# Function to stop vLLM server (for normal model transitions)
stop_vllm_server() {
    log_info "Stopping vLLM server..."
    
    # Method 1: Kill using PID file if it exists
    if [ -f vllm_server.pid ]; then
        local pid=$(cat vllm_server.pid)
        log_info "Found PID file with PID: $pid"
        
        if ps -p $pid > /dev/null 2>&1; then
            log_info "Killing process group for PID: $pid"
            # Kill the entire process group to catch all child processes
            kill -TERM -$pid 2>/dev/null || true
            sleep 3
            
            # Force kill if still running
            if ps -p $pid > /dev/null 2>&1; then
                log_warning "Process still running, force killing process group..."
                kill -KILL -$pid 2>/dev/null || true
                sleep 2
            fi
        fi
        rm -f vllm_server.pid
    fi
    
    # Method 2: Kill any remaining vLLM processes by name
    log_info "Cleaning up any remaining vLLM processes..."
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 2
    pkill -9 -f "vllm serve" 2>/dev/null || true
    
    # Method 3: Kill processes using the specific port
    log_info "Killing processes using port $VLLM_PORT..."
    local port_pids=$(lsof -ti:$VLLM_PORT 2>/dev/null || true)
    if [ -n "$port_pids" ]; then
        echo "$port_pids" | xargs -r kill -TERM 2>/dev/null || true
        sleep 2
        # Force kill if still there
        port_pids=$(lsof -ti:$VLLM_PORT 2>/dev/null || true)
        if [ -n "$port_pids" ]; then
            echo "$port_pids" | xargs -r kill -KILL 2>/dev/null || true
        fi
    fi
    
    # Method 4: Clear GPU memory (if nvidia-ml-py is available)
    if command -v nvidia-smi > /dev/null 2>&1; then
        log_info "Clearing GPU memory..."
        nvidia-smi --gpu-reset 2>/dev/null || true
    fi
    
    # Wait a moment for cleanup
    sleep 2
    
    # Verify server is stopped
    if curl -s -f "http://${VLLM_HOST}:${VLLM_PORT}/health" > /dev/null 2>&1; then
        log_warning "Server may still be running on port $VLLM_PORT"
    else
        log_success "vLLM server stopped successfully"
    fi
}

# Function to run evaluation
run_evaluation() {
    local model_name="$1"
    
    log_info "Running evaluation for model: $model_name"
    
    # Build the command arguments
    local eval_args="--model-name \"$model_name\""
    eval_args+=" --base-url \"$BASE_URL\""
    eval_args+=" --output-dir \"$OUTPUT_DIR\""
    eval_args+=" --task-name \"$TASK_NAME\""
    eval_args+=" --batch-size $BATCH_SIZE"
    eval_args+=" --num-fewshot $NUM_FEWSHOT"
    eval_args+=" --num-concurrent $NUM_CONCURRENT"
    eval_args+=" --max-retries $MAX_RETRIES"
    
    if [ -n "$LIMIT" ]; then
        eval_args+=" --limit $LIMIT"
    fi
    
    # Add log samples configuration
    if [ "$LOG_SAMPLES" = false ]; then
        eval_args+=" --no-log-samples"
    fi
    
    # Add tokenized requests if enabled
    if [ "$TOKENIZED_REQUESTS" = true ]; then
        eval_args+=" --tokenized-requests"
    fi
    
    log_info "Running: python $EVAL_SCRIPT $eval_args"
    
    # Run the evaluation
    if eval "python $EVAL_SCRIPT $eval_args"; then
        log_success "Evaluation completed for $model_name"
        return 0
    else
        log_error "Evaluation failed for $model_name"
        return 1
    fi
}

# Global flag to track if we're in controlled execution
SCRIPT_INTERRUPTED=false
SCRIPT_COMPLETED=false

# Function to handle interruption signals
handle_interrupt() {
    log_warning "Script interrupted! Cleaning up..."
    
    if [ -f vllm_server.pid ]; then
        local pid=$(cat vllm_server.pid)
        if ps -p $pid > /dev/null 2>&1; then
            log_info "Stopping vLLM server (PID: $pid)..."
            kill -TERM $pid 2>/dev/null || true
            sleep 3
            if ps -p $pid > /dev/null 2>&1; then
                kill -KILL $pid 2>/dev/null || true
            fi
        fi
        rm -f vllm_server.pid
    fi
    
    log_info "Cleanup completed"
    exit 1
}

# Set trap only for interruption
trap handle_interrupt INT TERM

# Main execution
main() {
    log_info "Starting batch evaluation of ${#MODELS[@]} models"
    log_info "Configuration:"
    log_info "  Task: $TASK_NAME"
    log_info "  Few-shot: $NUM_FEWSHOT"
    log_info "  Batch size: $BATCH_SIZE"
    log_info "  Output directory: $OUTPUT_DIR"
    log_info "  Base URL: $BASE_URL"
    log_info "  Server timeout: ${SERVER_READY_TIMEOUT}s"
    
    if [ -n "$LIMIT" ]; then
        log_info "  Sample limit: $LIMIT"
    fi
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Initialize summary
    local summary_file="$OUTPUT_DIR/batch_summary.txt"
    echo "Batch Evaluation Summary" > "$summary_file"
    echo "========================" >> "$summary_file"
    echo "Started: $(date)" >> "$summary_file"
    echo "Task: $TASK_NAME" >> "$summary_file"
    echo "Few-shot: $NUM_FEWSHOT" >> "$summary_file"
    echo "Batch size: $BATCH_SIZE" >> "$summary_file"
    echo "Models: ${#MODELS[@]}" >> "$summary_file"
    if [ -n "$LIMIT" ]; then
        echo "Sample limit: $LIMIT" >> "$summary_file"
    fi
    echo "" >> "$summary_file"

    # Loop through models
    for i in "${!MODELS[@]}"; do
        local model="${MODELS[$i]}"
        local model_num=$((i + 1))
        
        echo ""
        log_info "=========================================="
        log_info "Processing model $model_num/${#MODELS[@]}: $model"
        log_info "=========================================="
        
        # Start server
        if start_vllm_server "$model"; then
            # Run evaluation
            if run_evaluation "$model"; then
                log_success "Model $model completed successfully"
                echo "✅ $model - SUCCESS" >> "$summary_file"
            else
                log_error "Evaluation failed for $model"
                echo "❌ $model - EVALUATION FAILED" >> "$summary_file"
            fi
        else
            log_error "Failed to start server for $model"
            echo "❌ $model - SERVER FAILED" >> "$summary_file"
        fi
        
        # Stop server before next model
        log_info "Debugging"
        if [ $model_num -lt ${#MODELS[@]} ]; then
            log_info "Stopping server for current model (transitioning to next model)"
        else
            log_info "Stopping server for current model (evaluation complete)"
        fi
        stop_vllm_server
        
        # Brief pause between models
        if [ $model_num -lt ${#MODELS[@]} ]; then
            log_info "Preparing for next model in 5 seconds..."
            sleep 5
        fi
    done
    
    # Final summary
    echo "" >> "$summary_file"
    echo "Completed: $(date)" >> "$summary_file"
    echo "Failed: $failed" >> "$summary_file"
    
    echo ""
    log_info "=========================================="
    log_info "Batch evaluation completed!"

    log_info "Summary saved to: $summary_file"
    log_info "=========================================="
    
    # Display summary
    cat "$summary_file"
}

# Parse command line arguments (these can override the configuration above)
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            VLLM_PORT="$2"
            BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1/completions"
            shift 2
            ;;
        --task)
            TASK_NAME="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num-fewshot)
            NUM_FEWSHOT="$2"
            shift 2
            ;;
        --gpu-memory-utilization)
            GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        --tensor-parallel-size)
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --timeout)
            SERVER_READY_TIMEOUT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --port PORT                    vLLM server port (default: $VLLM_PORT)"
            echo "  --task TASK                    Evaluation task (default: $TASK_NAME)"
            echo "  --batch-size SIZE              Batch size (default: $BATCH_SIZE)"
            echo "  --num-fewshot N                Number of few-shot examples (default: $NUM_FEWSHOT)"
            echo "  --gpu-memory-utilization UTIL  GPU memory utilization (default: $GPU_MEMORY_UTILIZATION)"
            echo "  --tensor-parallel-size SIZE    Tensor parallel size (default: $TENSOR_PARALLEL_SIZE)"
            echo "  --limit N                      Limit number of samples for testing (default: no limit)"
            echo "  --output-dir DIR               Output directory (default: $OUTPUT_DIR)"
            echo "  --timeout SECONDS              Server ready timeout (default: ${SERVER_READY_TIMEOUT}s)"
            echo "  --help                         Show this help message"
            echo ""
            echo "Configuration:"
            echo "  Task: $TASK_NAME"
            echo "  Few-shot: $NUM_FEWSHOT"
            echo "  Batch size: $BATCH_SIZE"
            echo "  Server timeout: ${SERVER_READY_TIMEOUT}s"
            echo ""
            echo "Models to evaluate:"
            for model in "${MODELS[@]}"; do
                echo "  - $model"
            done
            echo ""
            echo "To customize the evaluation, edit the CONFIGURATION SECTION at the top of this script."
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if evaluation script exists
if [ ! -f "$EVAL_SCRIPT" ]; then
    log_error "Evaluation script not found: $EVAL_SCRIPT"
    exit 1
fi

# Run main function
main
