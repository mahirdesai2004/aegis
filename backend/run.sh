#!/usr/bin/env bash
# run.sh — Start Aegis FastAPI backend

echo "🚀 Starting Aegis Backend..."
echo "📍 API will be available at: http://0.0.0.0:10000"
echo "📚 Docs at: http://0.0.0.0:10000/docs"
echo ""

uvicorn app.main:app --host 0.0.0.0 --port 10000
