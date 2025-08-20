# Creates datasets sequentially by function type

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="/share/u/yu.stev/hops/dataset-generator/datasets/10.1/hops10.1plus1"

SEED_FILE="/share/u/yu.stev/hops/dataset-generator/seed/seeds_10F_1D_plus1.jsonl"

# Optional explicit seeds file to use; leave empty to auto-detect
SEED_FILE=""
if [ -z "$SEED_FILE" ]; then
    # Prefer latest seeds_*F_*D.jsonl, else fallback to seeds.jsonl
    LATEST_SEEDS=$(ls -t "$SCRIPT_DIR/../seed"/seeds_*F_*D.jsonl 2>/dev/null | head -n 1)
    if [ -n "$LATEST_SEEDS" ]; then
        SEED_FILE="$LATEST_SEEDS"
    elif [ -f "$SCRIPT_DIR/../seed/seeds.jsonl" ]; then
        SEED_FILE="$SCRIPT_DIR/../seed/seeds.jsonl"
    fi
fi
# Build seed arg once
SEED_ARG=""
if [ -n "$SEED_FILE" ]; then
    SEED_ARG="--seed-file \"$SEED_FILE\""
fi

FAMILIES="D"
# Option 1: Manually enumerate depths
DEPTHS="1"
# Option 2: Set MAX_DEPTH to auto-generate DEPTHS sequence 0..MAX_DEPTH
# MAX_DEPTH=2
if [ -n "$MAX_DEPTH" ]; then
    # shellcheck disable=SC2046
    DEPTHS="$(seq 0 "$MAX_DEPTH")"
fi

# Desired max datapoints per function per depth layer (0 means no cap)
# Per-depth overrides:
TARGET_PER_FUNC_DEPTH0=200
TARGET_PER_FUNC_DEPTH1=50
TARGET_PER_FUNC_DEPTH2=50
# Default cap for any unspecified depths (0 = no cap)
TARGET_PER_FUNC_DEPTH_DEFAULT=0

BASE_VARIATIONS=4
WRAPPER_VARIATIONS=9

PLUS_ONE=true

cap_file() {
    file_path="$1"
    max_count="$2"
    if [ -z "$max_count" ] || [ "$max_count" -le 0 ]; then
        return 0
    fi
    if [ ! -f "$file_path" ]; then
        return 0
    fi
    total_lines=$(wc -l < "$file_path" 2>/dev/null | tr -d ' ')
    if [ -z "$total_lines" ]; then
        return 0
    fi
    if [ "$total_lines" -gt "$max_count" ]; then
        tmp_file="${file_path}.tmp"
        # Randomly downsample to the requested count
        shuf "$file_path" | head -n "$max_count" > "$tmp_file" && mv "$tmp_file" "$file_path"
        echo "Capped $file_path from $total_lines to $max_count lines"
    fi
}

# Collect generated files for final combination
FILES_ARG=""

for fam in $FAMILIES; do
    # Base depth 0 (only if DEPTHS includes 0)
    if echo " $DEPTHS " | grep -q " 0 "; then
        base_out="$DATASET_DIR/${fam}0.jsonl"
        eval "python \"$SCRIPT_DIR/create_base_dataset.py\" --output-file \"$base_out\" \"<${fam}0>\" --variations-per-seed $BASE_VARIATIONS $SEED_ARG"
        cap_file "$base_out" "$TARGET_PER_FUNC_DEPTH0"
        FILES_ARG="$FILES_ARG \"$base_out\""
    fi

    # Wrapper depths > 0
    for d in $DEPTHS; do
        if [ "$d" != "0" ]; then
            wrap_out="$DATASET_DIR/hop_${fam}${d}.jsonl"
            if [ "$PLUS_ONE" = "true" ]; then
                eval "python \"$SCRIPT_DIR/create_wrapper_dataset.py\" --output-file \"$wrap_out\" --function \"<${fam}${d}>\" --variations-per-seed $WRAPPER_VARIATIONS $SEED_ARG --plus-one"
            else
                eval "python \"$SCRIPT_DIR/create_wrapper_dataset.py\" --output-file \"$wrap_out\" --function \"<${fam}${d}>\" --variations-per-seed $WRAPPER_VARIATIONS $SEED_ARG"
            fi
            # Choose cap based on depth-specific setting with default fallback
            cap_var_name="TARGET_PER_FUNC_DEPTH${d}"
            cap_value=$(eval echo \$$cap_var_name)
            if [ -z "$cap_value" ]; then
                cap_value="$TARGET_PER_FUNC_DEPTH_DEFAULT"
            fi
            cap_file "$wrap_out" "$cap_value"
            FILES_ARG="$FILES_ARG \"$wrap_out\""
        fi
    done
done

# Combine all generated (and possibly capped) files into one
COMBINED_OUTPUT="$DATASET_DIR/combined_all.jsonl"
echo "Combining datasets into $COMBINED_OUTPUT"
eval "python \"$SCRIPT_DIR/combine_datasets.py\" --input-files $FILES_ARG --output-file \"$COMBINED_OUTPUT\" --seed 42"