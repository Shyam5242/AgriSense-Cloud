@echo off
REM Copy to run_ingestion.bat (git ignores that name), set your project id, then schedule it with Windows Task Scheduler.
cd /d "%~dp0"
set GEE_PROJECT=your-earth-engine-project-id
call venv\Scripts\activate
python ingest_satellite.py >> ingestion_log.txt 2>&1
