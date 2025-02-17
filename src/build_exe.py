import sys
import os
import argparse
import subprocess
import mediapipe
import shutil

# Argument parser setup
parser = argparse.ArgumentParser(description="Build a Python script into an executable with PyInstaller.")
parser.add_argument("--file", required=True, help="The Python file to compile into an .exe")
args = parser.parse_args()

# Get the filename without extension
script_name = os.path.splitext(os.path.basename(args.file))[0]

# Get the Mediapipe installation path
mp_path = os.path.dirname(mediapipe.__file__)

# Define the required module paths
palm_detection_path = os.path.join(mp_path, "modules", "palm_detection")
hand_landmark_path = os.path.join(mp_path, "modules", "hand_landmark")

# Build the PyInstaller command dynamically
pyinstaller_cmd = [
    "pyinstaller",
    "--onefile",
    "--console",
    f'--add-data="{palm_detection_path};mediapipe/modules/palm_detection"',
    f'--add-data="{hand_landmark_path};mediapipe/modules/hand_landmark"',
    args.file,  # Use the file provided via command line argument
]

# Run PyInstaller
subprocess.run(" ".join(pyinstaller_cmd), shell=True)

# Clean up build/ folder and .spec file
build_folder = "build"
spec_file = f"{script_name}.spec"

if os.path.exists(build_folder) and os.path.exists(spec_file):
    shutil.rmtree(build_folder)  # Delete the build/ folder
    os.remove(spec_file)  # Delete the .spec file    

    print("\n✅ Build completed successfully!")
    print(f"📁 Your .exe file is in the 'dist/' folder: dist/{script_name}.exe")
    print("🧹 Cleaned up build files.")
else:
    print("\n❌ Build failed. Please check the error message above.") 
