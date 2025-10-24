#!/usr/bin/env python3
"""
Simple setup script for Sign Language Recognition System
Uses OpenCV instead of MediaPipe for Python 3.13 + ARM64 compatibility
"""

import os
import sys
import subprocess
import platform

def print_banner():
    """Print the project banner."""
    print("=" * 60)
    print("    Sign Language Recognition System - Simple Setup")
    print("=" * 60)
    print("OpenCV-based version for Python 3.13 + ARM64")
    print("=" * 60)

def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} - Compatible")
    return True

def install_simple_dependencies():
    """Install simplified dependencies."""
    print("\nInstalling simplified dependencies...")
    
    try:
        # Upgrade pip first
        print("Upgrading pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install simplified requirements
        print("Installing simplified requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_simple.txt"])
        
        print("✅ Dependencies installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    except FileNotFoundError:
        print("❌ requirements_simple.txt not found")
        return False

def create_virtual_environment():
    """Create a virtual environment."""
    print("\nCreating virtual environment...")
    
    venv_name = "venv"
    
    if os.path.exists(venv_name):
        print(f"✅ Virtual environment '{venv_name}' already exists")
        return True
    
    try:
        subprocess.check_call([sys.executable, "-m", "venv", venv_name])
        print(f"✅ Virtual environment '{venv_name}' created successfully")
        
        # Show activation instructions
        if platform.system() == "Windows":
            print(f"   Activate with: {venv_name}\\Scripts\\activate")
        else:
            print(f"   Activate with: source {venv_name}/bin/activate")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False

def test_imports():
    """Test if core packages can be imported."""
    print("\nTesting imports...")
    
    required_packages = [
        'cv2',
        'numpy',
        'pyttsx3',
        'flask',
        'flask_cors'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError as e:
            print(f"  ❌ {package} - {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed imports: {', '.join(failed_imports)}")
        return False
    
    print("✅ All imports successful!")
    return True

def create_directories():
    """Create necessary project directories."""
    print("\nCreating project directories...")
    
    directories = ['data', 'models', 'static', 'templates']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"  ✅ Created: {directory}/")
        else:
            print(f"  ✅ Exists: {directory}/")

def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    print("\nTesting basic functionality...")
    
    try:
        # Test if we can read the files
        with open('hand_tracking_opencv.py', 'r') as f:
            content = f.read()
            if 'class HandTrackerOpenCV' in content:
                print("  ✅ HandTrackerOpenCV class found")
            else:
                print("  ❌ HandTrackerOpenCV class not found")
                return False
        
        with open('sign_recognition_simple.py', 'r') as f:
            content = f.read()
            if 'class SimpleSignRecognizer' in content:
                print("  ✅ SimpleSignRecognizer class found")
            else:
                print("  ❌ SimpleSignRecognizer class not found")
                return False
        
        print("✅ Basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality tests failed: {e}")
        return False

def show_next_steps():
    """Show next steps for the user."""
    print("\n" + "=" * 60)
    print("🎉 Simple setup completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Activate virtual environment (if created):")
    if platform.system() == "Windows":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    
    print("\n2. Start the simplified system:")
    print("   python3 sign_recognition_simple.py")
    
    print("\n3. Or test hand tracking:")
    print("   python3 hand_tracking_opencv.py")
    
    print("\n4. For best results:")
    print("   - Ensure good lighting")
    print("   - Keep hands clearly visible")
    print("   - Use a quality webcam")
    print("   - Make clear hand gestures")
    
    print("\n5. What this version provides:")
    print("   - Basic hand detection using OpenCV")
    print("   - Simple gesture recognition")
    print("   - Audio feedback for recognized signs")
    print("   - Compatible with Python 3.13 + ARM64")
    
    print("\n6. Limitations:")
    print("   - Less accurate than MediaPipe version")
    print("   - Basic gesture recognition only")
    print("   - May need adjustment for your setup")
    
    print("\nHappy signing! 🤟")

def main():
    """Main setup function."""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Setup cannot continue. Please upgrade Python.")
        return
    
    # Create virtual environment
    create_virtual_environment()
    
    # Install simplified dependencies
    if not install_simple_dependencies():
        print("\n❌ Setup failed. Please check the error messages above.")
        return
    
    # Test imports
    if not test_imports():
        print("\n❌ Some packages failed to import. Please reinstall dependencies.")
        return
    
    # Create directories
    create_directories()
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n⚠️  Basic functionality tests failed, but setup completed.")
    
    # Show next steps
    show_next_steps()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
        print("You can run setup_simple.py again to complete the setup.")
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        print("Please check the error messages and try again.")
