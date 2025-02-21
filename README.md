# Hand Tracking 🖐️  
Control your mouse with your hand!  

## 🚀 Features  
✅ Hand-tracking for mouse control  
✅ Supports different modes (with/without window)  
✅ Easy installation and usage  
✅ Available as an executable (.exe)  

## 🛠️ System Requirements  
- **Operating System**: Windows (other OS may require adjustments)  
- **Python**: >=3.8, <3.12 (only required for Python users)  
- **Hardware**: A functional camera  

## 📥 Installation & Usage  

### 1️⃣ **Easiest Method: Use the Prebuilt Executables**  
Download the latest version from the [GitHub Releases](https://github.com/Hawk3388/hand-tracking/releases) page and run the program directly – no Python installation required! Prebuilt executables are available for **Windows, Linux, and macOS**.  

### 2️⃣ **Run Directly with Python**  
If you prefer to run the application using Python:  

#### 🔹 Clone the Repository  
```sh  
git clone https://github.com/Hawk3388/hand-tracking.git  
cd hand-tracking  
```  

#### 🔹 Install Dependencies  
```sh  
pip install -r requirements.txt  
```  

#### 🔹 Start the Application  
```sh  
python src/hand-tracking.py  
```  
If you don’t want a graphical window, use:  
```sh  
python src/hand-tracking-no-window.py  
```  

### 3️⃣ **Build Your Own `.exe`**  
If you want to create your own executable file, first install `pyinstaller`:  
```sh  
pip install pyinstaller==6.12.0  
```  
Then, create the `.exe` file using the following command:  
```sh  
python src/build_exe.py --file src/hand-tracking.py  # or src/hand-tracking-no-window.py  
```  

## 📝 License  
This project is licensed under the **MIT License**. See the [`LICENSE`](LICENSE) file for more details.
