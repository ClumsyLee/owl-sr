#!/usr/bin/env bash

python3 fetcher.py
python3 render.py
git commit -am 'Update data.'
git push
