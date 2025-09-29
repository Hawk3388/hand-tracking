import sys
import os
import subprocess
import mediapipe
import shutil
from pathlib import Path

# Detect OS
IS_WINDOWS = sys.platform.startswith("win")

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

# Check if PyTorch can be imported without MKL issues
try:
    import torch
    print("✅ PyTorch import successful")
except ImportError as e:
    print(f"⚠️  PyTorch import failed: {e}")
    print("💡 Consider installing PyTorch without MKL:")
    print("   conda install pytorch torchvision torchaudio cpuonly -c pytorch")

# Correct path separator based on OS
path_separator = ";" if IS_WINDOWS else ":"

def build_executable(script_path, script_name, console_mode="--console"):
    """Build a single executable"""
    print(f"\n🔨 Building {script_name}...")
    
    # Build the PyInstaller command dynamically - optimized for smaller file
    pyinstaller_cmd = [
        "pyinstaller",
        "--onefile",
        console_mode,  # Use console mode parameter
        "--exclude-module=pygame",      # Not needed
        "--exclude-module=cv2.cuda",    # CUDA not needed
        # Ensure numpy is properly bundled
        "--hidden-import=numpy",
        "--hidden-import=numpy._core._multiarray_tests",
        "--collect-all=numpy",
        f'--add-data="{palm_detection_path}{path_separator}mediapipe/modules/palm_detection"',
        f'--add-data="{hand_landmark_path}{path_separator}mediapipe/modules/hand_landmark"',
    ]
    
    # Add ASL modules if they exist
    if asl_exists:
        # Add the entire asl directory
        pyinstaller_cmd.append(f'--add-data="{asl_dir}{path_separator}asl"')
        print("✅ Added ASL module to build")
    
    # Add PyTorch hidden imports (minimal for ASL)
    pyinstaller_cmd.extend([
        "--hidden-import=torch",
        "--hidden-import=torch.nn",
        "--hidden-import=torch.nn.functional",
        "--hidden-import=torch.optim",
        "--hidden-import=torch.utils",
        "--hidden-import=torch.utils.data",
        # OpenCV and numpy dependencies
        "--hidden-import=cv2",
        "--hidden-import=pyautogui",
        "--hidden-import=mediapipe",
        "--hidden-import=matplotlib",  # Required by mediapipe drawing utils
        # Collect all torch data and binaries
        "--collect-all=torch",
        "--collect-data=torch",
    ])
    
    # Add the main script
    pyinstaller_cmd.append(script_path)
    
    # Run PyInstaller
    result = subprocess.run(" ".join(pyinstaller_cmd), shell=True)
    
    if result.returncode != 0:
        print(f"❌ Build failed for {script_name}")
        return False
    
    # Clean up build/ folder and .spec file
    build_folder = "build"
    spec_file = f"{script_name}.spec"
    
    if os.path.exists(build_folder):
        shutil.rmtree(build_folder)  # Delete the build/ folder
    
    if os.path.exists(spec_file):
        os.remove(spec_file)  # Delete the .spec file
    
    return True

# Build both versions - with option for smaller builds
scripts_to_build = [
    ("src/hand-tracking.py", "hand-tracking"),
    ("src/hand-tracking-no-window.py", "hand-tracking-no-window")
]

# Option for smaller builds without ASL
BUILD_WITH_ASL = True  # Set to False for smaller files without ASL

if not BUILD_WITH_ASL:
    print("⚠️  Building WITHOUT ASL support for smaller file size")
    asl_exists = False
    # Remove PyTorch dependencies for non-ASL builds
    print("🧹 Removing PyTorch dependencies for smaller build")

success_count = 0
for script_path, script_name in scripts_to_build:
    # Use --noconsole for GUI version, --console for background version
    console_mode = "--noconsole" if "no-window" not in script_name else "--console"
    if build_executable(script_path, script_name, console_mode):
        success_count += 1

# Show final results
print(f"\n{'='*50}")
if success_count == len(scripts_to_build):
    print("✅ All builds completed successfully!")
else:
    print(f"⚠️  {success_count}/{len(scripts_to_build)} builds completed")

# Check output files
output_ext = ".exe" if IS_WINDOWS else ""
for script_path, script_name in scripts_to_build:
    output_file = f"dist/{script_name}{output_ext}"
    if os.path.exists(output_file):
        print(f"📁 {script_name}: {output_file}")
    else:
        print(f"❌ {script_name}: Build failed")

print("\n🧹 Cleaned up build files.")

# Show usage instructions
print("\n📖 Usage Instructions:")
print("   • hand-tracking.exe: Full version with camera window")
print("   • hand-tracking-no-window.exe: Background version")
print("   • For ASL mode: Use Thumb+Middle finger to switch modes")
print("   • For mouse mode: Use hand gestures to control mouse")
print("   • Multi-monitor: Move hand to screen edges to switch monitors")

if model_exists:
    print("✅ ASL model included in builds")
else:
    print("⚠️  Warning: ASL model not found - ASL features may not work")
