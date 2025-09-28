import sys
import os
import argparse
import subprocess
import mediapipe
import shutil
from pathlib import Path

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

# ASL module paths
current_dir = Path(__file__).parent
asl_dir = current_dir.parent / "asl"
asl_model_dir = asl_dir / "model"

# Check if ASL files exist
asl_exists = asl_dir.exists() and (asl_dir / "model.py").exists()
model_exists = (asl_model_dir / "frame_mlp_asl.pt").exists()

print(f"ASL module found: {asl_exists}")
print(f"ASL model found: {model_exists}")

# Correct path separator based on OS
path_separator = ";" if IS_WINDOWS else ":"

# Build the PyInstaller command dynamically
pyinstaller_cmd = [
    "pyinstaller",
    "--onefile",
    "--console",
    f'--add-data="{palm_detection_path}{path_separator}mediapipe/modules/palm_detection"',
    f'--add-data="{hand_landmark_path}{path_separator}mediapipe/modules/hand_landmark"',
]

# Add ASL modules if they exist
if asl_exists:
    # Add the entire asl directory
    pyinstaller_cmd.append(f'--add-data="{asl_dir}{path_separator}asl"')
    print("✅ Added ASL module to build")

# # Add PyTorch hidden imports (required for PyTorch models)
# pyinstaller_cmd.extend([
#     "--hidden-import=torch",
#     "--hidden-import=torch.nn",
#     "--hidden-import=torch.nn.functional", 
#     "--hidden-import=torchvision",
#     "--collect-submodules=torch"
# ])

# Add the main script
pyinstaller_cmd.append(args.file)

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
    
    # Check if ASL model was included
    if model_exists:
        print("✅ ASL model included in build")
    else:
        print("⚠️  Warning: ASL model not found - ASL features may not work")
    
    print("🧹 Cleaned up build files.")
    
    # Show usage instructions
    print("\n📖 Usage Instructions:")
    print(f"   • Run: {output_file}")
    print("   • For ASL mode: Use Thumb+Middle finger to switch modes")
    print("   • For mouse mode: Use hand gestures to control mouse")
    
else:
    print("\n❌ Build failed. Please check the error message above.")
    if not asl_exists:
        print("💡 Note: ASL module not found - building without ASL support")
    if not model_exists:
        print("💡 Note: ASL model file not found - ASL features will be disabled")
