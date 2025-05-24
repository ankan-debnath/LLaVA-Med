#!/bin/bash


pip install uv
uv install
uv pip install --force-reinstall --no-cache-dir -r requirements.txt

uv pip  install -U transformers
