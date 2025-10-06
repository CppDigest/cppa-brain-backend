#!/usr/bin/env python3
"""
Test script to verify thread extraction functionality
"""
import os
import sys
from pathlib import Path

def test_basic_functionality():
    """Test basic functionality without full processing"""
    print("Testing basic functionality...")
    
    # Test imports
    try:
        import json
        import mailbox
        import re
        from typing import Dict, List, Any, Optional
        print("✓ All imports successful")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    # Test file access
    source_dir = Path("data/source_data/new/en")
    if not source_dir.exists():
        print(f"✗ Source directory not found: {source_dir}")
        return False
    print("✓ Source directory exists")
    
    # Test specific files
    threads_file = source_dir / "messages" / "mail_json" / "threads_boost_lists_boost_org.json"
    if not threads_file.exists():
        print(f"✗ Threads file not found: {threads_file}")
        return False
    print("✓ Threads file exists")
    
    emails_file = source_dir / "messages" / "mail_json" / "emails_boost_lists_boost_org.json"
    if not emails_file.exists():
        print(f"✗ Emails file not found: {emails_file}")
        return False
    print("✓ Emails file exists")
    
    mbox_file = source_dir / "messages" / "mbox" / "boost@lists.boost.org.mbox"
    if not mbox_file.exists():
        print(f"✗ Mbox file not found: {mbox_file}")
        return False
    print("✓ Mbox file exists")
    
    # Test JSON loading
    try:
        with open(threads_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"✓ Threads JSON loaded: {len(data.get('results', []))} threads")
    except Exception as e:
        print(f"✗ Error loading threads JSON: {e}")
        return False
    
    # Test mbox loading
    try:
        mbox = mailbox.mbox(mbox_file)
        print(f"✓ Mbox loaded: {len(mbox)} messages")
    except Exception as e:
        print(f"✗ Error loading mbox: {e}")
        return False
    
    print("\n✓ All basic tests passed!")
    return True

def test_small_extraction():
    """Test extraction with a small sample"""
    print("\nTesting small extraction...")
    
    try:
        from simple_thread_extractor import load_threads_data, load_emails_data, load_mbox_data
        
        source_dir = "data/source_data/new/en"
        
        # Load a small sample
        threads_file = f"{source_dir}/messages/mail_json/threads_boost_lists_boost_org.json"
        emails_file = f"{source_dir}/messages/mail_json/emails_boost_lists_boost_org.json"
        mbox_file = f"{source_dir}/messages/mbox/boost@lists.boost.org.mbox"
        
        print("Loading threads...")
        threads_data = load_threads_data(threads_file)
        print(f"Loaded {len(threads_data)} threads")
        
        print("Loading emails...")
        emails_data = load_emails_data(emails_file)
        print(f"Loaded {len(emails_data)} emails")
        
        print("Loading mbox (first 100 messages)...")
        # For testing, we'll limit mbox loading
        mbox_messages = {}
        try:
            mbox = mailbox.mbox(mbox_file)
            for i, message in enumerate(mbox):
                if i >= 100:  # Limit for testing
                    break
                message_id = message.get('Message-ID', '').strip()
                if message_id:
                    content = ""
                    if message.is_multipart():
                        for part in message.walk():
                            if part.get_content_type() == 'text/plain':
                                payload = part.get_payload(decode=True)
                                if payload:
                                    try:
                                        content = payload.decode('utf-8', errors='ignore')
                                    except UnicodeDecodeError:
                                        content = str(payload)
                                    break
                    else:
                        payload = message.get_payload(decode=True)
                        if payload:
                            try:
                                content = payload.decode('utf-8', errors='ignore')
                            except UnicodeDecodeError:
                                content = str(payload)
                    
                    mbox_messages[message_id] = {
                        'message_id': message_id,
                        'subject': message.get('Subject', ''),
                        'content': content,
                        'from': message.get('From', ''),
                        'date': message.get('Date', ''),
                        'to': message.get('To', ''),
                        'cc': message.get('Cc', ''),
                        'reply_to': message.get('Reply-To', '')
                    }
        except Exception as e:
            print(f"Error loading mbox: {e}")
            return False
            
        print(f"Loaded {len(mbox_messages)} mbox messages")
        
        # Test matching
        from simple_thread_extractor import match_messages_to_threads
        thread_messages = match_messages_to_threads(threads_data, emails_data, mbox_messages)
        print(f"Matched {len(thread_messages)} threads with messages")
        
        print("✓ Small extraction test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Small extraction test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Thread Extraction Test ===")
    
    if test_basic_functionality():
        test_small_extraction()
    else:
        print("Basic tests failed, skipping extraction test")
    
    print("\n=== Test completed ===")
