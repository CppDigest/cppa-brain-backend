#!/usr/bin/env python3
"""
Comprehensive test runner for thread extraction functionality
"""
import sys
import os
from pathlib import Path

def run_import_tests():
    """Test all imports work correctly"""
    print("Testing imports...")
    try:
        import json
        import mailbox
        import re
        from typing import Dict, List, Any, Optional
        from pathlib import Path
        from email.utils import parseaddr
        print("✓ All standard library imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def run_syntax_tests():
    """Test that all Python files have valid syntax"""
    print("\nTesting syntax...")
    python_files = [
        "extract_thread_data.py",
        "simple_thread_extractor.py", 
        "test_thread_extraction.py"
    ]
    
    for file in python_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                compile(f.read(), file, 'exec')
            print(f"✓ {file} syntax is valid")
        except SyntaxError as e:
            print(f"✗ {file} has syntax error: {e}")
            return False
        except Exception as e:
            print(f"✗ Error reading {file}: {e}")
            return False
    
    return True

def run_basic_functionality_tests():
    """Test basic functionality without full processing"""
    print("\nTesting basic functionality...")
    
    try:
        # Test that we can import our modules
        sys.path.insert(0, '.')
        
        # Test simple_thread_extractor functions
        from simple_thread_extractor import (
            create_output_directory,
            load_threads_data,
            load_emails_data,
            load_mbox_data,
            extract_message_content,
            extract_thread_id_from_url,
            match_messages_to_threads,
            save_thread_files,
            process_list
        )
        print("✓ All function imports successful")
        
        # Test URL extraction
        test_url = "http://lists.boost.org/archives/api/list/boost@lists.boost.org/thread/ABC123/"
        thread_id = extract_thread_id_from_url(test_url)
        if thread_id == "ABC123":
            print("✓ Thread ID extraction works")
        else:
            print(f"✗ Thread ID extraction failed: expected 'ABC123', got '{thread_id}'")
            return False
        
        # Test directory creation
        test_dir = "test_output"
        create_output_directory(test_dir)
        if Path(test_dir).exists():
            print("✓ Directory creation works")
            # Clean up
            import shutil
            shutil.rmtree(test_dir)
        else:
            print("✗ Directory creation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def run_file_structure_tests():
    """Test that all required files exist"""
    print("\nTesting file structure...")
    
    required_files = [
        "extract_thread_data.py",
        "simple_thread_extractor.py",
        "test_thread_extraction.py",
        "README.md",
        "requirements.txt",
        ".gitignore",
        "LICENSE",
        "setup.py",
        "pyproject.toml"
    ]
    
    for file in required_files:
        if Path(file).exists():
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} missing")
            return False
    
    return True

def run_documentation_tests():
    """Test that documentation is complete"""
    print("\nTesting documentation...")
    
    # Check README has required sections
    try:
        with open("README.md", 'r', encoding='utf-8') as f:
            readme_content = f.read()
        
        required_sections = [
            "# Thread Data Extraction",
            "## Overview",
            "## Usage",
            "## Requirements",
            "## License"
        ]
        
        for section in required_sections:
            if section in readme_content:
                print(f"✓ README contains: {section}")
            else:
                print(f"✗ README missing: {section}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Documentation test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Thread Extraction Test Suite ===")
    
    tests = [
        ("Import Tests", run_import_tests),
        ("Syntax Tests", run_syntax_tests),
        ("File Structure Tests", run_file_structure_tests),
        ("Documentation Tests", run_documentation_tests),
        ("Basic Functionality Tests", run_basic_functionality_tests),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running {test_name}...")
        print('='*50)
        
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    print('='*50)
    
    if passed == total:
        print("🎉 All tests passed! Ready for GitHub.")
        return True
    else:
        print("❌ Some tests failed. Please fix issues before pushing to GitHub.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
