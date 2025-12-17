#!/bin/bash

echo "🚀 Menyalakan Producer (Background)..."
python producer.py &

echo "⏳ Menunggu 5 detik..."
sleep 5

echo "🔥 Menyalakan Collector Spark (Background)..."
python collect_stream.py &

echo "🧠 Menyalakan Trainer (Background)..."
python train_model.py &

echo "📊 Menyalakan Streamlit (Foreground)..."
streamlit run stream_final.py --server.address=0.0.0.0