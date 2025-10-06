# Thread Data Extraction

This script extracts thread data from mbox and mail JSON files and saves them as individual thread JSON files.

## Overview

The extraction process combines data from three sources:
1. **Thread JSON files** - Contains thread metadata (subject, date, replies count, etc.)
2. **Emails JSON files** - Contains email metadata (message_id, thread_url, parent, children, sender_address, etc.)
3. **Mbox files** - Contains actual email content (subject, body, headers)

## File Structure

### Input Files
```
data/source_data/new/en/messages/
├── mail_json/
│   ├── threads_boost_lists_boost_org.json
│   ├── emails_boost_lists_boost_org.json
│   ├── threads_boost-announce_lists_boost_org.json
│   ├── emails_boost-announce_lists_boost_org.json
│   ├── threads_boost-users_lists_boost_org.json
│   └── emails_boost-users_lists_boost_org.json
└── mbox/
    ├── boost@lists.boost.org.mbox
    ├── boost-announce@lists.boost.org.mbox
    └── boost-users@lists.boost.org.mbox
```

### Output Files
```
data/processed/message_by_thread/
├── emails_boost_lists_boost_org/
│   ├── emails_boost_lists_boost_org_thread_0001.json
│   ├── emails_boost_lists_boost_org_thread_0002.json
│   └── ...
├── emails_boost-announce_lists_boost_org/
│   ├── emails_boost-announce_lists_boost_org_thread_0001.json
│   └── ...
└── emails_boost-users_lists_boost_org/
    ├── emails_boost-users_lists_boost_org_thread_0001.json
    └── ...
```

## Usage

### Basic Usage
```bash
python simple_thread_extractor.py
```

### Test First
```bash
python test_thread_extraction.py
```

## Output Format

Each thread JSON file contains:

```json
{
  "thread_info": {
    "url": "http://lists.boost.org/archives/api/list/boost@lists.boost.org/thread/...",
    "thread_id": "XJ2K26SJWZFNRIOTXLPJWAZGWGJBZJJF",
    "subject": "[boost] [container] New deque implementation and future plans",
    "date_active": "2025-10-02T09:17:00Z",
    "starting_email": "http://lists.boost.org/archives/api/list/boost@lists.boost.org/email/...",
    "emails_url": "http://lists.boost.org/archives/api/list/boost@lists.boost.org/thread/.../emails/",
    "replies_count": 9,
    "votes_total": 0
  },
  "messages": [
    {
      "message_id": "<message-id@example.com>",
      "subject": "[boost] [container] New deque implementation and future plans",
      "content": "Full email content here...",
      "thread_url": "http://lists.boost.org/archives/api/list/boost@lists.boost.org/thread/...",
      "parent": null,
      "children": ["<child-message-id@example.com>"],
      "sender_address": "sender@example.com",
      "from": "Sender Name <sender@example.com>",
      "date": "Wed, 02 Oct 2025 09:17:00 +0000",
      "to": "boost@lists.boost.org",
      "cc": "",
      "reply_to": "",
      "url": "http://lists.boost.org/archives/api/list/boost@lists.boost.org/email/..."
    }
  ],
  "message_count": 10
}
```

## Data Sources

### From Thread JSON:
- `url`, `thread_id`, `subject`, `date_active`, `starting_email`, `emails_url`, `replies_count`, `votes_total`

### From Emails JSON:
- `message_id`, `thread_url`, `parent`, `children`, `sender_address`, `date`, `url`

### From Mbox:
- `subject`, `content`, `from`, `date`, `to`, `cc`, `reply_to`

## Features

- **Thread-based organization**: Messages are grouped by thread
- **Complete data merging**: Combines metadata from all sources
- **Chronological ordering**: Threads and messages are sorted by date
- **Error handling**: Robust error handling for missing or corrupted files
- **Progress tracking**: Shows progress during processing

## Requirements

- Python 3.6+
- Standard library only (no external dependencies)

## File Naming Convention

Thread files are named as:
```
emails_{list_name}_thread_{number:04d}.json
```

Where:
- `list_name` is the mailing list name with special characters replaced
- `number` is the thread number (padded to 4 digits)
- Files are ordered by thread date (newest first)

## Error Handling

The script handles:
- Missing files gracefully
- Corrupted JSON files
- Encoding issues in mbox files
- Memory limitations for large files
- Network timeouts (if applicable)

## Performance Notes

- Large mbox files may take time to process
- Memory usage scales with file size
- Progress is shown for long operations
- Consider processing one list at a time for very large datasets

## Scripts

- `extract_thread_data.py` - Full-featured extractor with progress tracking
- `simple_thread_extractor.py` - Simplified version with basic error handling
- `test_thread_extraction.py` - Test script to verify functionality

## License

This project is licensed under the MIT License - see the LICENSE file for details.
