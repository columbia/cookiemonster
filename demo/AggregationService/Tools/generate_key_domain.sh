#!/bin/bash

JAR_FILE="avro-tools-1.11.1.jar"
SCHEMA_FILE="../data/output_domain.avsc"
INPUT_FILE="../data/output_domain_value.json"
DOMAIN_FILE="../data/output_domain.avro"

java -jar "$JAR_FILE" fromjson \
  --schema-file "$SCHEMA_FILE" \
  "$INPUT_FILE" > "$DOMAIN_FILE"
