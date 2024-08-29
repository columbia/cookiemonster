#!/bin/bash

JAR_FILE="LocalTestingTool_2.7.0.jar"
INPUT_DATA_FILE="../data/output_debug_clear_text_reports.avro"
DOMAIN_FILE="../data/output_domain.avro"
OUTPUT_DIRECTORY="../data"

java -jar "$JAR_FILE" \
  --input_data_avro_file "$INPUT_DATA_FILE" \
  --domain_avro_file "$DOMAIN_FILE" \
  --json_output \
  --output_directory "$OUTPUT_DIRECTORY" \
  # --no_noising
