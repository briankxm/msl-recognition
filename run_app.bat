@echo off
echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Starting Streamlit MSL Recognition App...
streamlit run app/app.py

pause
