@echo off
REM Change directory to the location of your project

REM Check if Python is installed
python --version
IF %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not added to PATH.
    exit /b 1
)

python -m venv aircraft
echo Virtual environment created.

call aircraft\Scripts\activate

pip install -r requirements.txt

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo Virtual environment setup complete.
pause