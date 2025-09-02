#!/usr/bin/env bash

# Use first argument as the virtual environment directory, default to ".venv"
ENV_DIR="${1:-.venv}"

whereis bash
whereis python
# Activate the chosen virtual environment
source "$ENV_DIR/bin/activate"
whereis python
