@echo off
echo ========================================
echo Adaptive Quality System Setup
echo ========================================
echo.

echo Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    echo Make sure Python and pip are installed
    pause
    exit /b 1
)

echo.
echo ========================================
echo Generating synthetic training data...
echo ========================================
python generate_training_data.py --output training_data/synthetic_performance_data.csv --samples 10000
if errorlevel 1 (
    echo ERROR: Failed to generate training data
    pause
    exit /b 1
)

echo.
echo ========================================
echo Training ML model...
echo ========================================
python train_adaptive_quality.py --data training_data/synthetic_performance_data.csv --output models/adaptive_quality.model
if errorlevel 1 (
    echo ERROR: Failed to train model
    pause
    exit /b 1
)

echo.
echo ========================================
echo Testing model...
echo ========================================
python test_model.py --model models/adaptive_quality.model

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo The adaptive quality system is ready to use!
echo.
echo Next steps:
echo   1. Build PlasmaDX-Clean (Debug or Release)
echo   2. Run the application
echo   3. Open ImGui panel (F1)
echo   4. Expand "Adaptive Quality (ML)" section
echo   5. Check "Enable Adaptive Quality"
echo   6. Select target FPS (120 recommended)
echo.
echo For real hardware optimization:
echo   - Enable "Collect Performance Data"
echo   - Run various test scenarios for 5-10 minutes
echo   - Run: python train_adaptive_quality.py --data training_data/performance_data.csv
echo.
pause
