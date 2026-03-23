@echo off
echo ========================================
echo Predictive Maintenance App
echo ========================================
echo.
echo Используется Anaconda Python...
echo.

cd /d C:\Users\Hacker\Desktop\predictive_maintenance_project

REM Используем Anaconda Python
C:\Users\Hacker\anaconda3\python.exe -m streamlit run app.py

pause