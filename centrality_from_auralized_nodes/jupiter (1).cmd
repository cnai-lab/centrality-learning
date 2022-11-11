CD /D "C:\ProgramData\Anaconda3\Scripts\"
conda init cmd.exe
CALL conda.bat activate
CD /D "%~dp0"
"C:\ProgramData\Anaconda3\Scripts\jupyter" notebook --notebook-dir="."
