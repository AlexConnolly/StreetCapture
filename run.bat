@echo off
REM StreetCapture launcher. Uses the project venv.
REM   run.bat                                  -> webcam (index 0)
REM   run.bat --source "rtsp://user:pass@IP:554/stream1"   -> Tapo / RTSP
REM   run.bat --headless                       -> no windows, data only
call "%~dp0.venv\Scripts\activate.bat"
python -m streetcapture %*
