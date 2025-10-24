#!/usr/bin/env python3
"""
Basic functionality test for Sign Language Recognition System
Tests core components without external dependencies.
"""

import os
import sys
import importlib.util

def test_file_structure():
    """Test if all required files exist."""
    print("Testing project file structure...")
    
    required_files = [
        'README.md',
        'requirements.txt',
        'hand_tracking.py',
        'sign_recognition.py',
        'app.py',
        'demo.py',
        'train_model.py',
        'quick_start.py',
        'setup.py'
    ]
    
    required_dirs = [
        'data',
        'models',
        'static',
        'templates'
    ]
    
    all_good = True
    
    # Check files
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - MISSING")
            all_good = False
    
    # Check directories
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"  ✅ {directory}/")
        else:
            print(f"  ❌ {directory}/ - MISSING")
            all_good = False
    
    return all_good

def test_python_syntax():
    """Test if Python files have valid syntax."""
    print("\nTesting Python syntax...")
    
    python_files = [
        'hand_tracking.py',
        'sign_recognition.py',
        'app.py',
        'demo.py',
        'train_model.py',
        'quick_start.py',
        'setup.py'
    ]
    
    all_good = True
    
    for file in python_files:
        try:
            with open(file, 'r') as f:
                compile(f.read(), file, 'exec')
            print(f"  ✅ {file} - Valid syntax")
        except SyntaxError as e:
            print(f"  ❌ {file} - Syntax error: {e}")
            all_good = False
        except Exception as e:
            print(f"  ❌ {file} - Error: {e}")
            all_good = False
    
    return all_good

def test_imports():
    """Test if core modules can be imported (without external deps)."""
    print("\nTesting module imports...")
    
    # Test basic imports that don't require external packages
    try:
        import os
        import sys
        import json
        import time
        import threading
        print("  ✅ Standard library imports")
    except ImportError as e:
        print(f"  ❌ Standard library import error: {e}")
        return False
    
    # Test if we can read the files
    try:
        with open('hand_tracking.py', 'r') as f:
            content = f.read()
            if 'class HandTracker' in content:
                print("  ✅ HandTracker class found")
            else:
                print("  ❌ HandTracker class not found")
                return False
    except Exception as e:
        print(f"  ❌ Error reading hand_tracking.py: {e}")
        return False
    
    try:
        with open('sign_recognition.py', 'r') as f:
            content = f.read()
            if 'class SignRecognizer' in content:
                print("  ✅ SignRecognizer class found")
            else:
                print("  ❌ SignRecognizer class not found")
                return False
    except Exception as e:
        print(f"  ❌ Error reading sign_recognition.py: {e}")
        return False
    
    return True

def test_requirements():
    """Test requirements.txt file."""
    print("\nTesting requirements.txt...")
    
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read().strip().split('\n')
        
        if len(requirements) > 0:
            print(f"  ✅ {len(requirements)} dependencies listed")
            for req in requirements[:5]:  # Show first 5
                print(f"    - {req}")
            if len(requirements) > 5:
                print(f"    ... and {len(requirements) - 5} more")
            return True
        else:
            print("  ❌ No dependencies listed")
            return False
    except Exception as e:
        print(f"  ❌ Error reading requirements.txt: {e}")
        return False

def test_templates():
    """Test HTML template file."""
    print("\nTesting HTML template...")
    
    template_file = 'templates/index.html'
    
    if os.path.exists(template_file):
        try:
            with open(template_file, 'r') as f:
                content = f.read()
                
            # Check for key elements
            checks = [
                ('HTML structure', '<!DOCTYPE html>' in content),
                ('JavaScript', '<script>' in content),
                ('CSS styling', '<style>' in content),
                ('Sign language title', 'Sign Language Recognition' in content),
                ('Video feed', 'video_feed' in content)
            ]
            
            all_checks_passed = True
            for check_name, passed in checks:
                if passed:
                    print(f"    ✅ {check_name}")
                else:
                    print(f"    ❌ {check_name}")
                    all_checks_passed = False
            
            return all_checks_passed
        except Exception as e:
            print(f"  ❌ Error reading template: {e}")
            return False
    else:
        print("  ❌ Template file not found")
        return False

def test_configuration():
    """Test configuration and settings."""
    print("\nTesting configuration...")
    
    # Check if directories exist and are writable
    test_dirs = ['data', 'models']
    
    for directory in test_dirs:
        if os.path.exists(directory):
            try:
                # Try to create a test file
                test_file = os.path.join(directory, '.test_write')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                print(f"  ✅ {directory}/ - Writable")
            except Exception as e:
                print(f"  ❌ {directory}/ - Not writable: {e}")
                return False
        else:
            print(f"  ❌ {directory}/ - Does not exist")
            return False
    
    return True

def main():
    """Run all tests."""
    print("Sign Language Recognition System - Basic Tests")
    print("=" * 50)
    print("Running basic functionality tests...")
    print("(These tests don't require external dependencies)")
    print()
    
    tests = [
        ("File Structure", test_file_structure),
        ("Python Syntax", test_python_syntax),
        ("Module Imports", test_imports),
        ("Requirements", test_requirements),
        ("HTML Template", test_templates),
        ("Configuration", test_configuration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"Result: {'✅ PASSED' if result else '❌ FAILED'}")
        except Exception as e:
            print(f"Error: {e}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All basic tests passed! The project structure is correct.")
        print("Next steps:")
        print("1. Install dependencies: python3 setup.py")
        print("2. Start the system: python3 quick_start.py")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the issues above.")
        print("The project may not work correctly until these issues are resolved.")
    
    return passed == total

if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)
