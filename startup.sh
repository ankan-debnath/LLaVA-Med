#!/bin/bash

pip install uv
uv venv venv

uv pip install --force-reinstall --no-cache-dir -r requirements.txt

