#!/bin/bash

ARCHIVE_NAME="evaluation_results.tar.gz"

echo "Packaging JSON outputs..."

# Create a tar.gz file containing the three JSON output files
# This preserves the path structure, which is safer
tar -czf $ARCHIVE_NAME \
    outputs_simple/llama3.2/sequential/sequential_risk_regulatory_data_consumer.json \
    outputs_simple/llama3.2/sequential/sequential_consumer_data_regulatory_risk.json \
    outputs_simple/llama3.2/sequential/sequential_data_consumer_regulatory_risk.json

echo "--------------------------------------------------------"
echo "✅ Archive created: $ARCHIVE_NAME"
echo ""
echo "To download to your local machine, run this command from your LOCAL terminal:"
echo "scp $(whoami)@$(hostname):$(pwd)/$ARCHIVE_NAME ."
echo "--------------------------------------------------------"
