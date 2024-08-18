#!/bin/bash

if [ -d "tb_logs" ]; then
    rm -r tb_logs/*
elif [ -d "src/tb_logs" ]; then
    rm -r src/tb_logs/*
elif [ -d "../src/tb_logs" ]; then
    rm -r ../src/tb_logs/*
else
    echo "Error: tb_logs directory not found."
    exit 1
fi