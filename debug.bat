@echo off
clear

Rem Build
if not exist Build\Debug mkdir Build\Debug
cd Build\Debug
cmake.exe -S ..\..\ -B . -G"MinGW Makefiles" -D CMAKE_EXPORT_COMPILE_COMMANDS=ON -D CMAKE_BUILD_TYPE=Debug
C:\MinGW\mingw32\bin\mingw32-make.exe
if not exist ..\compile_commands.json move compile_commands.json ..\

Rem Lauch
MyProject.exe
cd ../../

