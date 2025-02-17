# Hand-tracking
Control your mouse with your hand

# System requirements
* python >=3.8, <3.12
*  a camara

# Installation
clone the repository
```sh
git clone https://github.com/Hawk3388/hand-tracking
cd hand-tracking
```
now run
```sh
pip install -r requirements.txt
```
then 
```sh
python src/hand-tracking.py
```
or
```sh
python src/hand-tracking-no-window.py
```
if you don't want a window

# Usage
Download the latest build from the [GitHub releases](https://github.com/Hawk3388/han-tracking/releases) page.

If you want to build your own .exe file, you have to install `pyinstaller`
```sh
pip install pyinstaller==6.12.0
```
to run
```sh
python src/build_exe.py --file hand-tracking.py (or hand-tracking-no-window.py
```
