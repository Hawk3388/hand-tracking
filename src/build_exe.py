import sys
import os
import argparse
import subprocess
import mediapipe
import shutil

# Detect OS
IS_WINDOWS = sys.platform.startswith("win")

# Argument parser setup
parser = argparse.ArgumentParser(description="Build a Python script into an executable with PyInstaller.")
parser.add_argument("--file", required=True, help="The Python file to compile into an executable")
args = parser.parse_args()

# Get the filename without extension
script_name = os.path.splitext(os.path.basename(args.file))[0]

# Get the Mediapipe installation path
mp_path = os.path.dirname(mediapipe.__file__)

# Define the required module paths
palm_detection_path = os.path.join(mp_path, "modules", "palm_detection")
hand_landmark_path = os.path.join(mp_path, "modules", "hand_landmark")

# Correct path separator based on OS
path_separator = ";" if IS_WINDOWS else ":"

# Build the PyInstaller command dynamically
pyinstaller_cmd = [
    "pyinstaller",
    "--onefile",
    "--console",
    f'--add-data="{palm_detection_path}{path_separator}mediapipe/modules/palm_detection"',
    f'--add-data="{hand_landmark_path}{path_separator}mediapipe/modules/hand_landmark"',
    args.file,
]

# Run PyInstaller
subprocess.run(" ".join(pyinstaller_cmd), shell=True)

# Clean up build/ folder and .spec file
build_folder = "build"
spec_file = f"{script_name}.spec"

if os.path.exists(build_folder):
    shutil.rmtree(build_folder)  # Delete the build/ folder

if os.path.exists(spec_file):
    os.remove(spec_file)  # Delete the .spec file    

# Show success message
output_ext = ".exe" if IS_WINDOWS else ""  # Linux executables have no extension
output_file = f"dist/{script_name}{output_ext}"

if os.path.exists(output_file):
    print("\n✅ Build completed successfully!")
    print(f"📁 Your executable is in the 'dist/' folder: {output_file}")
    print("🧹 Cleaned up build files.")
else:
    print("\n❌ Build failed. Please check the error message above.")
