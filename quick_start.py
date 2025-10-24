#!/usr/bin/env python3
"""
Quick Start Script for Sign Language Recognition System
This script provides easy access to the main functionality.
"""

import os
import sys
import subprocess
import webbrowser
import time

def check_dependencies():
    """Check if required packages are installed."""
    print("Checking dependencies...")
    
    required_packages = [
        'opencv-python',
        'mediapipe', 
        'tensorflow',
        'numpy',
        'pyttsx3',
        'flask',
        'flask-cors'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("All dependencies are installed!")
    return True

def create_directories():
    """Create necessary directories."""
    print("\nCreating project directories...")
    
    directories = ['data', 'models', 'static', 'templates']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"  Created: {directory}/")
        else:
            print(f"  Exists: {directory}/")

def start_web_interface():
    """Start the Flask web interface."""
    print("\nStarting web interface...")
    print("The web interface will open in your browser automatically.")
    print("Press Ctrl+C to stop the server.")
    
    try:
        # Start Flask app in background
        process = subprocess.Popen([sys.executable, 'app.py'])
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Open browser
        webbrowser.open('http://localhost:5000')
        
        print("Web interface started successfully!")
        print("Server is running at: http://localhost:5000")
        
        # Wait for user to stop
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\nStopping server...")
            process.terminate()
            process.wait()
            print("Server stopped.")
            
    except Exception as e:
        print(f"Error starting web interface: {e}")
        print("You can manually start it by running: python app.py")

def start_command_line():
    """Start the command-line interface."""
    print("\nStarting command-line interface...")
    print("Press Ctrl+C to stop.")
    
    try:
        subprocess.run([sys.executable, 'sign_recognition.py'])
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"Error starting command-line interface: {e}")

def run_demo():
    """Run the demo script."""
    print("\nStarting demo...")
    print("This will show you the basic functionality of the system.")
    
    try:
        subprocess.run([sys.executable, 'demo.py'])
    except Exception as e:
        print(f"Error running demo: {e}")

def train_model():
    """Start the model training process."""
    print("\nStarting model training...")
    print("This will guide you through collecting data and training a custom model.")
    
    try:
        subprocess.run([sys.executable, 'train_model.py'])
    except Exception as e:
        print(f"Error starting training: {e}")

def show_help():
    """Show help information."""
    print("\nSign Language Recognition System - Quick Start")
    print("=" * 50)
    print("This system provides real-time sign language recognition with audio output.")
    print()
    print("Available options:")
    print("1. Web Interface - Modern web-based interface (recommended)")
    print("2. Command Line - Direct command-line interface")
    print("3. Demo - Interactive demonstration of features")
    print("4. Train Model - Collect data and train custom model")
    print("5. Check Setup - Verify system configuration")
    print("6. Help - Show this help message")
    print("7. Exit")
    print()
    print("For best results:")
    print("- Ensure good lighting")
    print("- Keep hands clearly visible in camera")
    print("- Use a good quality webcam")
    print("- Hold signs steady for accurate recognition")

def main():
    """Main quick start function."""
    print("Sign Language Recognition System")
    print("Quick Start Script")
    print("=" * 40)
    
    # Check dependencies first
    if not check_dependencies():
        print("\nPlease install missing dependencies and try again.")
        return
    
    # Create directories
    create_directories()
    
    while True:
        print("\n" + "=" * 40)
        print("Quick Start Menu")
        print("=" * 40)
        print("1. Start Web Interface")
        print("2. Start Command Line Interface")
        print("3. Run Demo")
        print("4. Train Custom Model")
        print("5. Check System Setup")
        print("6. Help")
        print("7. Exit")
        
        try:
            choice = input("\nSelect an option (1-7): ").strip()
            
            if choice == '1':
                start_web_interface()
            elif choice == '2':
                start_command_line()
            elif choice == '3':
                run_demo()
            elif choice == '4':
                train_model()
            elif choice == '5':
                print("\nSystem Setup Check:")
                print("=" * 20)
                check_dependencies()
                create_directories()
                print("\nSystem is ready to use!")
            elif choice == '6':
                show_help()
            elif choice == '7':
                print("\nExiting...")
                break
            else:
                print("Invalid choice. Please select 1-7.")
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")

if __name__ == '__main__':
    main()
