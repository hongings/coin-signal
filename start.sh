#!/bin/bash
python notifier.py &
streamlit run app.py \
  --server.port $PORT \
  --server.address 0.0.0.0 \
  --server.headless true
