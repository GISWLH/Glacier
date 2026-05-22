@echo off
cd /d E:\Glacier
call D:\miniconda3\Scripts\activate.bat gee
set HTTP_PROXY=
set HTTPS_PROXY=
set GIT_HTTP_PROXY=
set GIT_HTTPS_PROXY=
python .\src\submit_gee_batch_region_year.py --region alaska --year 2024 --status-filter all --max-tasks 5 --runlist-csv .\data\prepared\execution_plan\alaska_2024_anomaly_recheck_remaining5_runlist.csv --name-suffix _recheck_m6789
pause
