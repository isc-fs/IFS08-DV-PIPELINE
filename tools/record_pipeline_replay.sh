#!/bin/bash
# Record Pipeline Replay - Capture sensor inputs AND pipeline outputs
#
# This script:
# 1. Plays back an existing rosbag (sensor data)
# 2. Runs the DV pipeline with mission injector
# 3. Records a NEW bag with BOTH inputs and pipeline outputs
#
# Usage:
#   ./record_pipeline_replay.sh <source_bag> <mission_name> [output_name]
#
# Example:
#   cd ~/dv_ws/src/IFS08-DV-PIPELINE/tools
#   ./record_pipeline_replay.sh ~/bags/autocross_track_carparity trackdrive

# Get script directory for finding mission_injector.py
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Arguments
SOURCE_BAG="$1"
MISSION_NAME="${2:-trackdrive}"
OUTPUT_NAME="$3"

# Auto-generate output name if not provided
if [ -z "$OUTPUT_NAME" ]; then
    BASENAME=$(basename "$SOURCE_BAG")
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_NAME="bench_${BASENAME}_${TIMESTAMP}"
fi

OUTPUT_PATH="$HOME/bags/${OUTPUT_NAME}"

# Validate source bag
if [ -z "$SOURCE_BAG" ]; then
    echo -e "${RED}Error: No source rosbag specified${NC}"
    echo "Usage: $0 <source_bag> [mission_name] [output_name]"
    echo ""
    echo "Example:"
    echo "  $0 ~/bags/autocross_track_carparity trackdrive"
    exit 1
fi

if [ ! -d "$SOURCE_BAG" ]; then
    echo -e "${RED}Error: Source bag not found: $SOURCE_BAG${NC}"
    exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Pipeline Replay Recording${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Source bag:    $SOURCE_BAG"
echo "Mission:       $MISSION_NAME"
echo "Output:        $OUTPUT_PATH"
echo ""

# Create temp directory for logs and PIDs
TMPDIR=$(mktemp -d)
echo "Temp dir: $TMPDIR"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo -e "${YELLOW}Cleaning up...${NC}"
    
    # Kill processes
    if [ -f "$TMPDIR/recorder.pid" ]; then
        RECORDER_PID=$(cat "$TMPDIR/recorder.pid")
        if kill -0 $RECORDER_PID 2>/dev/null; then
            echo "Stopping recorder..."
            kill -SIGINT $RECORDER_PID
            sleep 2
        fi
    fi
    
    if [ -f "$TMPDIR/injector.pid" ]; then
        INJECTOR_PID=$(cat "$TMPDIR/injector.pid")
        if kill -0 $INJECTOR_PID 2>/dev/null; then
            echo "Stopping mission injector..."
            kill $INJECTOR_PID
        fi
    fi
    
    if [ -f "$TMPDIR/pipeline.pid" ]; then
        PIPELINE_PID=$(cat "$TMPDIR/pipeline.pid")
        if kill -0 $PIPELINE_PID 2>/dev/null; then
            echo "Stopping pipeline..."
            kill $PIPELINE_PID
            sleep 3
        fi
    fi
    
    # Show logs if there were errors
    if [ $? -ne 0 ]; then
        echo -e "${RED}Errors detected. Last 20 lines of logs:${NC}"
        echo "--- Pipeline ---"
        tail -20 "$TMPDIR/pipeline.log" 2>/dev/null || echo "No pipeline log"
        echo "--- Injector ---"
        tail -20 "$TMPDIR/injector.log" 2>/dev/null || echo "No injector log"
        echo "--- Recorder ---"
        tail -20 "$TMPDIR/recorder.log" 2>/dev/null || echo "No recorder log"
    fi
    
    rm -rf "$TMPDIR"
    echo -e "${GREEN}Done.${NC}"
}

trap cleanup EXIT INT TERM

# Step 1: Launch pipeline
echo -e "${YELLOW}[1/4] Launching DV pipeline...${NC}"
cd ~/dv_ws
source install/setup.bash

ros2 launch bringup sim_pipeline.launch.py \
    mission_name:="$MISSION_NAME" \
    use_sim_time:=true \
    > "$TMPDIR/pipeline.log" 2>&1 &

PIPELINE_PID=$!
echo $PIPELINE_PID > "$TMPDIR/pipeline.pid"
echo "Pipeline PID: $PIPELINE_PID"

echo "Waiting for pipeline to initialize (20s)..."
sleep 20

if ! kill -0 $PIPELINE_PID 2>/dev/null; then
    echo -e "${RED}Error: Pipeline failed to start!${NC}"
    tail -50 "$TMPDIR/pipeline.log"
    exit 1
fi

echo -e "${GREEN}✓ Pipeline running${NC}"
echo ""

# Step 2: Launch mission injector
echo -e "${YELLOW}[2/4] Launching mission injector...${NC}"

"$SCRIPT_DIR/mission_injector.py" \
    --mission "$MISSION_NAME" \
    --auto-drive \
    > "$TMPDIR/injector.log" 2>&1 &

INJECTOR_PID=$!
echo $INJECTOR_PID > "$TMPDIR/injector.pid"
echo "Mission injector PID: $INJECTOR_PID"

sleep 5

if ! kill -0 $INJECTOR_PID 2>/dev/null; then
    echo -e "${RED}Error: Mission injector failed!${NC}"
    tail -50 "$TMPDIR/injector.log"
    exit 1
fi

echo -e "${GREEN}✓ Mission injector running${NC}"
echo ""

# Step 3: Start recording OUTPUT bag (ALL TOPICS)
echo -e "${YELLOW}[3/4] Starting recording...${NC}"

# Record ALL topics with -a flag
# This captures:
#  - Original sensor data from manual bag (full lidar, IMU, motor, steering, etc.)
#  - All pipeline outputs (perception, planning, control, etc.)
#  - State topics, transforms, debug info, lifecycle events
#  - Everything else on the ROS graph

# Create output directory
mkdir -p "$HOME/bags"

# Start recorder
echo "Recording to: $OUTPUT_PATH"
echo "Recording ALL topics (-a flag)"
ros2 bag record \
    -a \
    --storage mcap \
    --output "$OUTPUT_PATH" \
    > "$TMPDIR/recorder.log" 2>&1 &

RECORDER_PID=$!
echo $RECORDER_PID > "$TMPDIR/recorder.pid"
echo "Recorder PID: $RECORDER_PID"

# Wait for recorder to initialize
sleep 3

if ! kill -0 $RECORDER_PID 2>/dev/null; then
    echo -e "${RED}Error: Recorder failed to start!${NC}"
    tail -50 "$TMPDIR/recorder.log"
    exit 1
fi

echo -e "${GREEN}✓ Recorder running${NC}"
echo ""

# Step 4: Play source bag
echo -e "${YELLOW}[4/4] Starting source bag playback...${NC}"

ros2 bag play "$SOURCE_BAG" \
    --clock \
    --rate 1.0 \
    > "$TMPDIR/player.log" 2>&1 &

PLAYER_PID=$!

echo "Playing bag... (this may take a while)"
echo ""

# Wait for playback to finish
wait $PLAYER_PID
PLAY_EXIT=$?

if [ $PLAY_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓ Playback complete${NC}"
else
    echo -e "${RED}✗ Playback failed (exit code: $PLAY_EXIT)${NC}"
    tail -50 "$TMPDIR/player.log"
fi

echo ""
echo "Waiting 5s for final messages to flush..."
sleep 5

# Stop recorder gracefully
echo "Stopping recorder..."
kill -SIGINT $RECORDER_PID 2>/dev/null
sleep 3

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Recording Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Output bag: $OUTPUT_PATH"

# Show bag info
if [ -d "$OUTPUT_PATH" ]; then
    BAG_SIZE=$(du -sh "$OUTPUT_PATH" | cut -f1)
    echo "Size: $BAG_SIZE"
    echo ""
    echo "To inspect:"
    echo "  ros2 bag info $OUTPUT_PATH"
    echo ""
    echo "To replay:"
    echo "  ros2 bag play $OUTPUT_PATH --clock"
fi
