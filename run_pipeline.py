import os
import sys
import unittest
import time

from src.utils import load_config, ensure_directories, setup_logger
from src.train import execute_full_training_pipeline
from src.evaluate import perform_benchmark_comparison
from generate_visualizations import generate_all_visualizations
from generate_report import build_pdf_report

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

logger = setup_logger("run_pipeline")

def main():
    print("=" * 70)
    print("SOLAR PV DEFECT DETECTION & HEALTH ASSESSMENT - FULL PIPELINE")
    print("=" * 70)

    config_path = "config/config.yaml"
    config = load_config(config_path)
    ensure_directories(config)

    # 1. Execute Training & Optuna HPO
    print("\n[STEP 1/5] Executing Model Training & Optuna HPO Study...")
    cnn_model, xgb_model, history, hpo_study, (train_df, val_df, test_df) = execute_full_training_pipeline(config)

    # 2. Benchmark Evaluation
    print("\n[STEP 2/5] Performing SOTA Benchmark Comparative Evaluation...")
    img_dir = config["dataset"]["real_data_dir"]
    img_dir = os.path.join(img_dir, "images")
    ensemble_metrics, bench_df, y_test, ensemble_preds, ensemble_probs = perform_benchmark_comparison(
        cnn_model, xgb_model, test_df, img_dir, config
    )

    # 3. Generate Analytical Visualizations
    print("\n[STEP 3/5] Generating 6 Analytical Visualization Figures...")
    generate_all_visualizations(config, cnn_model, xgb_model, test_df, history=history)

    # 4. Run Unit & Integration Tests
    print("\n[STEP 4/5] Running Unit & Integration Tests...")
    loader = unittest.TestLoader()
    suite = loader.discover('tests')
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    
    if not result.wasSuccessful():
        print("[WARNING] Some unit tests failed. Check logs.")
    else:
        print("[SUCCESS] All Unit & Integration Tests Passed Successfully!")

    # 5. Build IEEE PDF Report
    print("\n[STEP 5/5] Compiling 10-Page IEEE Formatted Report PDF...")
    report_path = config['paths']['report_output']
    build_pdf_report(pdf_filename=report_path, viz_dir=config['paths']['visualizations_dir'])

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Report PDF: {os.path.abspath(report_path)}")
    print(f"Figures:    {os.path.abspath(config['paths']['visualizations_dir'])}")
    print(f"Models:     {os.path.abspath(config['paths']['models_dir'])}")
    print("=" * 70)

if __name__ == "__main__":
    main()
