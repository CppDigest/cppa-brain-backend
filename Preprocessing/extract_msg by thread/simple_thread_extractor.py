#!/usr/bin/env python3
"""
Simplified thread data extractor with better error handling
"""
import json
import mailbox
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

def normalize_message_id(message_id: str) -> str:
    """Normalize Message-ID by stripping angle brackets and surrounding whitespace"""
    if not message_id:
        return ""
    mid = message_id.strip()
    # Remove surrounding angle brackets if present
    if mid.startswith('<') and mid.endswith('>'):
        mid = mid[1:-1].strip()
    return mid

def create_output_directory(output_dir: str):
    """Create output directory if it doesn't exist"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {output_dir}")

def load_threads_data(threads_file: str) -> Dict[str, Any]:
    """Load thread metadata from JSON file"""
    print(f"Loading threads from {threads_file}")
    threads_data = {}
    
    try:
        with open(threads_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for thread in data.get('results', []):
            thread_id = thread.get('thread_id')
            if thread_id:
                threads_data[thread_id] = {
                    'url': thread.get('url'),
                    'thread_id': thread_id,
                    'subject': thread.get('subject'),
                    'date_active': thread.get('date_active'),
                    'starting_email': thread.get('starting_email'),
                    'emails_url': thread.get('emails'),
                    'replies_count': thread.get('replies_count', 0),
                    'votes_total': thread.get('votes_total', 0)
                }
        print(f"Loaded {len(threads_data)} threads")
        
    except Exception as e:
        print(f"Error loading threads: {e}")
        raise
        
    return threads_data

def load_emails_data(emails_file: str) -> Dict[str, Any]:
    """Load email metadata from JSON file"""
    print(f"Loading emails from {emails_file}")
    emails_data = {}
    
    try:
        with open(emails_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for email in data.get('results', []):
            message_id = normalize_message_id(email.get('message_id', ''))
            if message_id:
                emails_data[message_id] = {
                    'message_id': message_id,
                    # Some datasets use 'thread' instead of 'thread_url'
                    'thread_url': email.get('thread_url') or email.get('thread'),
                    'parent': email.get('parent'),
                    'children': email.get('children', []),
                    'sender_address': email.get('sender_address'),
                    'date': email.get('date'),
                    'subject': email.get('subject'),
                    'url': email.get('url')
                }
        print(f"Loaded {len(emails_data)} emails")
        
    except Exception as e:
        print(f"Error loading emails: {e}")
        raise
        
    return emails_data

def extract_message_content(message) -> str:
    """Extract text content from email message"""
    content = ""
    
    try:
        if message.is_multipart():
            for part in message.walk():
                if part.get_content_type() == 'text/plain':
                    payload = part.get_payload(decode=True)
                    if payload:
                        try:
                            content += payload.decode('utf-8', errors='ignore')
                        except UnicodeDecodeError:
                            content += str(payload)
        else:
            payload = message.get_payload(decode=True)
            if payload:
                try:
                    content = payload.decode('utf-8', errors='ignore')
                except UnicodeDecodeError:
                    content = str(payload)
    except Exception as e:
        print(f"Error extracting content: {e}")
                    
    return content

def load_mbox_data(mbox_file: str) -> Dict[str, Any]:
    """Load message content from mbox file"""
    print(f"Loading mbox from {mbox_file}")
    mbox_messages = {}
    
    try:
        mbox = mailbox.mbox(mbox_file)
        
        for i, message in enumerate(mbox):
            if i % 1000 == 0:
                print(f"Processed {i} messages...")
                
            message_id = normalize_message_id(message.get('Message-ID', ''))
            if message_id:
                content = extract_message_content(message)
                
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
                
        print(f"Loaded {len(mbox_messages)} messages from mbox")
        
    except Exception as e:
        print(f"Error loading mbox: {e}")
        raise
        
    return mbox_messages

def extract_thread_id_from_url(url: str) -> Optional[str]:
    """Extract thread ID from thread URL"""
    if not url:
        return None
        
    # Pattern: /thread/THREAD_ID/
    match = re.search(r'/thread/([^/]+)/', url)
    if match:
        return match.group(1)
    return None

def match_messages_to_threads(threads_data: Dict, emails_data: Dict, mbox_messages: Dict) -> Dict[str, List[Dict]]:
    """Match messages to threads and create thread-based data structure"""
    print("Matching messages to threads...")
    
    thread_messages = {}
    
    # Group emails by thread
    for message_id, email_data in emails_data.items():
        thread_url = email_data.get('thread_url')
        if thread_url:
            thread_id = extract_thread_id_from_url(thread_url)
            if thread_id and thread_id in threads_data:
                if thread_id not in thread_messages:
                    thread_messages[thread_id] = []
                
                # Get mbox content if available
                mbox_data = mbox_messages.get(message_id, {})
                
                # Combine email metadata with mbox content
                combined_message = {
                    'message_id': message_id,
                    'subject': mbox_data.get('subject', email_data.get('subject', '')),
                    'content': mbox_data.get('content', ''),
                    'thread_url': thread_url,
                    'parent': email_data.get('parent'),
                    'children': email_data.get('children', []),
                    'sender_address': email_data.get('sender_address'),
                    'from': mbox_data.get('from', ''),
                    'date': mbox_data.get('date', email_data.get('date', '')),
                    'to': mbox_data.get('to', ''),
                    'cc': mbox_data.get('cc', ''),
                    'reply_to': mbox_data.get('reply_to', ''),
                    'url': email_data.get('url')
                }
                
                thread_messages[thread_id].append(combined_message)
    
    print(f"Matched messages to {len(thread_messages)} threads")
    return thread_messages

def save_thread_files(thread_messages: Dict[str, List[Dict]], threads_data: Dict[str, Any], 
                     list_name: str, output_dir: str):
    """Save thread data to individual JSON files"""
    print(f"Saving thread files for {list_name}")
    
    # Create output directory for this list
    list_output_dir = Path(output_dir) / f"emails_{list_name.replace('@', '_').replace('.', '_')}"
    list_output_dir.mkdir(exist_ok=True)
    
    # Sort threads by date_active
    sorted_threads = sorted(
        threads_data.items(),
        key=lambda x: x[1].get('date_active', ''),
        reverse=True
    )
    
    saved_count = 0
    for i, (thread_id, thread_info) in enumerate(sorted_threads, 1):
        if thread_id in thread_messages:
            # Sort messages within thread by date
            messages = sorted(
                thread_messages[thread_id],
                key=lambda x: x.get('date', '')
            )
            
            # Create thread data structure
            thread_data = {
                'thread_info': thread_info,
                'messages': messages,
                'message_count': len(messages)
            }
            
            # Save to file
            filename = f"emails_{list_name.replace('@', '_').replace('.', '_')}_thread_{i:04d}.json"
            filepath = list_output_dir / filename
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(thread_data, f, indent=2, ensure_ascii=False)
                    
                print(f"Saved thread {i}: {filename} ({len(messages)} messages)")
                saved_count += 1
                
            except Exception as e:
                print(f"Error saving thread {i}: {e}")
    
    print(f"Saved {saved_count} thread files to {list_output_dir}")

def process_list(list_name: str, source_dir: str, output_dir: str):
    """Process a complete mailing list"""
    print(f"\n=== Processing {list_name} ===")
    
    # File paths
    threads_file = Path(source_dir) / "messages" / "mail_json" / f"threads_{list_name.replace('@', '_').replace('.', '_')}.json"
    emails_file = Path(source_dir) / "messages" / "mail_json" / f"emails_{list_name.replace('@', '_').replace('.', '_')}.json"
    mbox_file = Path(source_dir) / "messages" / "mbox" / f"{list_name}.mbox"

    print(f"Using threads file: {threads_file}")
    print(f"Using emails file: {emails_file}")
    print(f"Using mbox file: {mbox_file}")
    
    # Check if files exist
    if not threads_file.exists():
        print(f"Threads file not found: {threads_file}")
        return
    if not emails_file.exists():
        print(f"Emails file not found: {emails_file}")
        return
    if not mbox_file.exists():
        print(f"Mbox file not found: {mbox_file}")
        return
    
    try:
        # Load data
        threads_data = load_threads_data(threads_file)
        emails_data = load_emails_data(emails_file)
        mbox_messages = load_mbox_data(mbox_file)
        
        # Match messages to threads
        thread_messages = match_messages_to_threads(threads_data, emails_data, mbox_messages)
        
        # Save thread files
        save_thread_files(thread_messages, threads_data, list_name, output_dir)
        
    except Exception as e:
        print(f"Error processing {list_name}: {e}")
        raise

def main():
    """Main function to process all mailing lists"""
    source_dir = "data/source_data/new/en"
    output_dir = "data/processed/message_by_thread"
    
    # Create output directory
    create_output_directory(output_dir)
    
    # Mailing lists to process
    mailing_lists = [
        "boost@lists.boost.org",
        "boost-announce@lists.boost.org", 
        "boost-users@lists.boost.org"
    ]
    
    for list_name in mailing_lists:
        try:
            process_list(list_name, source_dir, output_dir)
        except Exception as e:
            print(f"Error processing {list_name}: {e}")
            continue
    
    print("\n=== Thread extraction completed ===")

if __name__ == "__main__":
    main()
