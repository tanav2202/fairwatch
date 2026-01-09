#!/bin/bash

ARCHIVE_NAME="gpu_evaluation_results.tar.gz"

echo "Packaging GPU results..."

# Create a tar.gz file containing the outputs from all 3 GPU folders and the log files
# We use find to safely handle cases where files might not exist yet
find outputs_gpu0 outputs_gpu1 outputs_gpu2 -name "*.json" > file_list.txt
find . -name "gpu*.log" >> file_list.txt

if [ ! -s file_list.txt ]; then
    echo "No result files or logs found yet."
    rm file_list.txt
    exit 1
fi

tar -czf $ARCHIVE_NAME -T file_list.txt

echo "--------------------------------------------------------"
echo "✅ Archive created: $ARCHIVE_NAME"
echo ""
echo "To download to your local machine, run this command from your LOCAL terminal:"
echo "scp $(whoami)@$(hostname):$(pwd)/$ARCHIVE_NAME ."
echo "--------------------------------------------------------"

rm file_list.txt
