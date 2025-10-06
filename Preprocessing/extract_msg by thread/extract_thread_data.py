#!/usr/bin/env python3
"""
Script to extract thread data from mbox and mail JSON files
and save as individual thread JSON files.
"""
import json
import mailbox
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from email.utils import parseaddr
import hashlib

class ThreadDataExtractor:
    def __init__(self, source_dir: str, output_dir: str):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.threads_data = {}
        self.emails_data = {}
        self.mbox_messages = {}
        
    def load_threads_data(self, threads_file: str):
        """Load thread metadata from JSON file"""
        print(f"Loading threads from {threads_file}")
        try:
            with open(threads_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            results = data.get('results', [])
            total_threads = len(results)
            print(f"Processing {total_threads} threads...")
            
            for i, thread in enumerate(results):
                if i % 1000 == 0 or i == total_threads - 1:
                    progress = (i + 1) / total_threads * 100
                    print(f"  Threads progress: {progress:.1f}% ({i + 1}/{total_threads})")
                    
                thread_id = thread.get('thread_id')
                if thread_id:
                    self.threads_data[thread_id] = {
                        'url': thread.get('url'),
                        'thread_id': thread_id,
                        'subject': thread.get('subject'),
                        'date_active': thread.get('date_active'),
                        'starting_email': thread.get('starting_email'),
                        'emails_url': thread.get('emails'),
                        'replies_count': thread.get('replies_count', 0),
                        'votes_total': thread.get('votes_total', 0)
                    }
            print(f"✓ Loaded {len(self.threads_data)} threads (100.0%)")
            
        except Exception as e:
            print(f"❌ Error loading threads: {e}")
            raise
        
    def load_emails_data(self, emails_file: str):
        """Load email metadata from JSON file"""
        print(f"Loading emails from {emails_file}")
        try:
            with open(emails_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            results = data.get('results', [])
            total_emails = len(results)
            print(f"Processing {total_emails} emails...")
            
            for i, email in enumerate(results):
                if i % 5000 == 0 or i == total_emails - 1:
                    progress = (i + 1) / total_emails * 100
                    print(f"  Emails progress: {progress:.1f}% ({i + 1}/{total_emails})")
                    
                message_id = email.get('message_id')
                if message_id:
                    self.emails_data[message_id] = {
                        'message_id': message_id,
                        'thread_url': email.get('thread'),
                        'parent': email.get('parent'),
                        'children': email.get('children', []),
                        'sender_address': email.get('sender_address'),
                        'date': email.get('date'),
                        'subject': email.get('subject'),
                        'url': email.get('url')
                    }
            print(f"✓ Loaded {len(self.emails_data)} emails (100.0%)")
            
        except Exception as e:
            print(f"❌ Error loading emails: {e}")
            raise
        
    def load_mbox_data(self, mbox_file: str):
        """Load message content from mbox file"""
        print(f"Loading mbox from {mbox_file}")
        try:
            mbox = mailbox.mbox(mbox_file)
            
            total_messages = len(mbox)
            print(f"Processing {total_messages} mbox messages...")
            
            for i, message in enumerate(mbox):
                if i % 1000 == 0 or i == total_messages - 1:
                    progress = (i + 1) / total_messages * 100
                    print(f"  Mbox progress: {progress:.1f}% ({i + 1}/{total_messages})")
                    
                message_id = message.get('Message-ID', '').strip()
                # Remove angle brackets from message_id
                if message_id.startswith('<') and message_id.endswith('>'):
                    message_id = message_id[1:-1]
                if message_id:
                    # Extract content
                    content = self._extract_message_content(message)
                    
                    self.mbox_messages[message_id] = {
                        'message_id': message_id,
                        'subject': message.get('Subject', ''),
                        'content': content,
                        'from': message.get('From', ''),
                        'date': message.get('Date', ''),
                        'to': message.get('To', ''),
                        'cc': message.get('Cc', ''),
                        'reply_to': message.get('Reply-To', '')
                    }
                    
            print(f"✓ Loaded {len(self.mbox_messages)} messages from mbox (100.0%)")
            
        except Exception as e:
            print(f"❌ Error loading mbox: {e}")
            raise
        
    def _extract_message_content(self, message) -> str:
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
            print(f"Warning: Error extracting content from message: {e}")
                    
        return content
        
    def match_messages_to_threads(self):
        """Match messages to threads and create thread-based data structure"""
        print("Matching messages to threads...")
        
        # Create thread-based structure
        thread_messages = {}
        total_emails = len(self.emails_data)
        processed = 0
        
        # Group emails by thread
        for message_id, email_data in self.emails_data.items():
            if processed % 5000 == 0 or processed == total_emails - 1:
                progress = (processed + 1) / total_emails * 100
                print(f"  Matching progress: {progress:.1f}% ({processed + 1}/{total_emails})")
                
            thread_url = email_data.get('thread_url')
            if thread_url:
                # Extract thread_id from URL
                thread_id = self._extract_thread_id_from_url(thread_url)
                if thread_id and thread_id in self.threads_data:
                    if thread_id not in thread_messages:
                        thread_messages[thread_id] = []
                    
                    # Get mbox content if available
                    mbox_data = self.mbox_messages.get(message_id, {})
                    
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
            
            processed += 1
        
        print(f"✓ Matched messages to {len(thread_messages)} threads (100.0%)")
        return thread_messages
        
    def _extract_thread_id_from_url(self, url: str) -> Optional[str]:
        """Extract thread ID from thread URL"""
        if not url:
            return None
            
        # Pattern: /thread/THREAD_ID/
        match = re.search(r'/thread/([^/]+)/', url)
        if match:
            return match.group(1)
        return None
        
    def save_thread_files(self, thread_messages: Dict[str, List[Dict]], list_name: str):
        """Save thread data to individual JSON files"""
        print(f"Saving thread files for {list_name}")
        
        # Create output directory for this list
        list_output_dir = self.output_dir / f"emails_{list_name.replace('@', '_').replace('.', '_')}"
        list_output_dir.mkdir(exist_ok=True)
        
        # Sort threads by date_active
        sorted_threads = sorted(
            self.threads_data.items(),
            key=lambda x: x[1].get('date_active', ''),
            reverse=True
        )
        
        total_threads = len([t for t in sorted_threads if t[0] in thread_messages])
        saved_count = 0
        
        for i, (thread_id, thread_info) in enumerate(sorted_threads, 1):
            if thread_id in thread_messages:
                if saved_count % 100 == 0 or saved_count == total_threads - 1:
                    progress = (saved_count + 1) / total_threads * 100
                    print(f"  Saving progress: {progress:.1f}% ({saved_count + 1}/{total_threads})")
                
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
                filename = f"emails_{list_name.replace('@', '_').replace('.', '_')}_thread_{i:05d}.json"
                filepath = list_output_dir / filename
                
                try:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(thread_data, f, indent=2, ensure_ascii=False)
                        
                    saved_count += 1
                except Exception as e:
                    print(f"❌ Error saving thread {i}: {e}")
                    continue
        
        print(f"✓ Saved {saved_count} thread files to {list_output_dir} (100.0%)")
        
    def process_list(self, list_name: str):
        """Process a complete mailing list"""
        print(f"\n=== Processing {list_name} ===")
        
        # File paths
        threads_file = self.source_dir / "messages" / "mail_json" / f"threads_{list_name.replace('@', '_').replace('.', '_')}.json"
        emails_file = self.source_dir / "messages" / "mail_json" / f"emails_{list_name.replace('@', '_').replace('.', '_')}.json"
        mbox_file = self.source_dir / "messages" / "mbox" / f"{list_name}.mbox"
        
        # Check if files exist
        if not threads_file.exists():
            print(f"❌ Threads file not found: {threads_file}")
            return
        if not emails_file.exists():
            print(f"❌ Emails file not found: {emails_file}")
            return
        if not mbox_file.exists():
            print(f"❌ Mbox file not found: {mbox_file}")
            return
        
        try:
            print("📋 Step 1/4: Loading thread metadata...")
            self.load_threads_data(threads_file)
            
            print("📧 Step 2/4: Loading email metadata...")
            self.load_emails_data(emails_file)
            
            print("📦 Step 3/4: Loading mbox content...")
            self.load_mbox_data(mbox_file)
            
            print("🔗 Step 4/4: Matching and saving threads...")
            thread_messages = self.match_messages_to_threads()
            self.save_thread_files(thread_messages, list_name)
            
            print(f"✅ Completed {list_name}")
            
        except Exception as e:
            print(f"❌ Error processing {list_name}: {e}")
            raise
        finally:
            # Clear data for next list
            self.threads_data.clear()
            self.emails_data.clear()
            self.mbox_messages.clear()

def main():
    """Main function to process all mailing lists"""
    source_dir = "data/source_data/new/en"
    output_dir = "data/processed/message_by_thread"
    
    # Mailing lists to process
    mailing_lists = [
        "boost@lists.boost.org",
        "boost-announce@lists.boost.org", 
        "boost-users@lists.boost.org"
    ]
    
    extractor = ThreadDataExtractor(source_dir, output_dir)
    
    print(f"🚀 Starting thread extraction for {len(mailing_lists)} mailing lists")
    print("=" * 60)
    
    for i, list_name in enumerate(mailing_lists, 1):
        overall_progress = i / len(mailing_lists) * 100
        print(f"\n📊 Overall Progress: {overall_progress:.1f}% ({i}/{len(mailing_lists)})")
        
        try:
            extractor.process_list(list_name)
        except Exception as e:
            print(f"❌ Error processing {list_name}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("🎉 Thread extraction completed successfully!")

if __name__ == "__main__":
    main()
