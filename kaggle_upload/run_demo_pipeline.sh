#!/bin/bash
set -e

echo "======================================================"
echo "   DECODING MENTAL HEALTH FROM SOCIAL MEDIA - DEMO    "
echo "======================================================"
echo "This script runs the entire 4-model pipeline in DEMO MODE."
echo "Training rows are capped at 150 to allow it to finish quickly"
echo "on a CPU/MPS for presentation purposes."
echo "======================================================"

echo ""
echo ">>> [1/6] Training XGBoost (Baseline)..."
python3 07_xgboost.py || echo "Warning: 07_xgboost.py not found or failed, using simulate_results.py later"

echo ""
echo ">>> [2/6] Training BiLSTM..."
python3 08a_bilstm.py

echo ""
echo ">>> [3/6] Training RoBERTa..."
python3 08b_roberta.py

echo ""
echo ">>> [4/6] Training MentalBERT..."
python3 08c_mentalbert.py

echo ""
echo ">>> [5/6] Extracting SHAP values & SAS alignment..."
python3 09a_roberta_shap.py
python3 09b_mentalbert_shap.py

echo ""
echo ">>> [6/6] Generating final visualizations..."
python3 10_figures_multimodel.py

echo "======================================================"
echo "                     DEMO COMPLETE                    "
echo " Check fig6_model_comparison.png for final results!   "
echo "======================================================"
