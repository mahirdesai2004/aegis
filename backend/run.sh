#!/bin/bash
# run.sh — Start Aegis FastAPI backend

cd "$(dirname "$0")"

echo "🚀 Starting Aegis Backend..."
echo "📍 API will be available at: http://localhost:8000"
echo "📚 Docs at: http://localhost:8000/docs"
echo ""

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
