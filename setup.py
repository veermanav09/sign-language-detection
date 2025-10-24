#!/usr/bin/env python3
"""
Setup script for Sign Language Recognition System
"""

import os
import sys
import subprocess
import platform

def print_banner():
    """Print the project banner."""
    print("=" * 60)
    print("    Sign Language Recognition System - Setup")
    print("=" * 60)
    print("Real-time sign language recognition with audio output")
    print("Built with MediaPipe, TensorFlow, and Flask")
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

def install_dependencies():
    """Install required dependencies."""
    print("\nInstalling dependencies...")
    
    try:
        # Upgrade pip first
        print("Upgrading pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        print("Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        print("✅ Dependencies installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    except FileNotFoundError:
        print("❌ requirements.txt not found")
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
    """Test if all required packages can be imported."""
    print("\nTesting imports...")
    
    required_packages = [
        'cv2',
        'mediapipe',
        'tensorflow',
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

def run_basic_tests():
    """Run basic functionality tests."""
    print("\nRunning basic tests...")
    
    try:
        # Test hand tracking
        print("  Testing hand tracking...")
        from hand_tracking import HandTracker
        tracker = HandTracker()
        print("    ✅ HandTracker initialized")
        tracker.release()
        
        # Test sign recognition
        print("  Testing sign recognition...")
        from sign_recognition import SignRecognizer
        recognizer = SignRecognizer()
        print("    ✅ SignRecognizer initialized")
        recognizer.hand_tracker.release()
        
        print("✅ Basic tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Basic tests failed: {e}")
        return False

def show_next_steps():
    """Show next steps for the user."""
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Activate virtual environment (if created):")
    if platform.system() == "Windows":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    
    print("\n2. Start the system:")
    print("   python quick_start.py")
    
    print("\n3. Or run specific components:")
    print("   python app.py              # Web interface")
    print("   python sign_recognition.py # Command line")
    print("   python demo.py             # Demo")
    print("   python train_model.py      # Training")
    
    print("\n4. For best results:")
    print("   - Ensure good lighting")
    print("   - Keep hands clearly visible")
    print("   - Use a quality webcam")
    print("   - Hold signs steady")
    
    print("\n5. Documentation:")
    print("   README.md - Project overview")
    print("   requirements.txt - Dependencies")
    
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
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed. Please check the error messages above.")
        return
    
    # Test imports
    if not test_imports():
        print("\n❌ Some packages failed to import. Please reinstall dependencies.")
        return
    
    # Create directories
    create_directories()
    
    # Run basic tests
    if not run_basic_tests():
        print("\n⚠️  Basic tests failed, but setup completed. You may need to troubleshoot.")
    
    # Show next steps
    show_next_steps()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
        print("You can run setup.py again to complete the setup.")
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        print("Please check the error messages and try again.")
