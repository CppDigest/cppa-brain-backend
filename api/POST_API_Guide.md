# POST API Request, Response, and Error Types Guide

## Overview
API requests are used for creating resources or sending data to a server. Unlike simple queries, API requests include a request body that contains the data to be sent to the server.

## API Request Structure

### Basic Structure
```json
{
  "key": "value",
  "data": "example"
}
```

### Common Request Fields
- `timestamp`: Request timestamp
- `requestId`: Unique request identifier
- `data`: The main payload data
- `auth`: Authentication credentials (if required)

## JSON Request Format

### Type 1: Email Thread Request
Request to get email thread information and messages.

**Endpoint:** `/api/maillist/thread/new`

**Request Example:**
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "requestId": "req_thread_12345",
  "type": "NEW THREAD",
  "data": {
    "thread_info": {
    "url": "http://lists.boost.org/archives/api/list/boost@lists.boost.org/thread/6UL7DQGCTGUYBLD4UCT2ZUKURBCJHHPT/",
    "thread_id": "6UL7DQGCTGUYBLD4UCT2ZUKURBCJHHPT",
    "subject": "[boost] Overload resolution speed",
    "date_active": "2011-09-25T10:38:08Z",
    "starting_email": "http://lists.boost.org/archives/api/list/boost@lists.boost.org/email/6UL7DQGCTGUYBLD4UCT2ZUKURBCJHHPT/",
    "emails_url": "http://lists.boost.org/archives/api/list/boost@lists.boost.org/thread/6UL7DQGCTGUYBLD4UCT2ZUKURBCJHHPT/emails/",
    "replies_count": 14,
    "votes_total": 0
    },
    "messages": [
      {
        "message_id": "j5hkfg$msc$1@dough.gmane.org",
        "subject": "Re: [boost] Overload resolution speed",
        "content": "On 22/09/2011 00:42, John Bytheway wrote:\n\n> What about using functors; so the call would be something like\n>\n> dispatching<f0_>()(h49_<int>(), h49_<int>(), h49_<int>(), h49_<int>(),\n> h49_<int>());\n>\n> with the f parameter choosing a template specialization, and then the\n> other parameters picking an overload of operator() ion it?\n\nThe problem is that you cannot overload dispatching<f0_>::operator() \nonce the class has been defined.\n\n\n",
        "thread_url": "http://lists.boost.org/archives/api/list/boost@lists.boost.org/thread/6UL7DQGCTGUYBLD4UCT2ZUKURBCJHHPT/",
        "parent": "http://lists.boost.org/archives/api/list/boost@lists.boost.org/email/6NPNBS36CWRJCGAWBRDC4YZJSV2HFG4B/",
        "children": [],
        "sender_address": "mathias.gaunard@ens-lyon.org",
        "from": "Mathias Gaunard <mathias.gaunard@ens-lyon.org>",
        "date": "Fri, 23 Sep 2011 11:45:19 +0200",
        "to": "boost@lists.boost.org",
        "cc": "",
        "reply_to": "",
        "url": "http://lists.boost.org/archives/api/list/boost@lists.boost.org/email/YINRZWKYACLGPDC7P67X6NZ4Y6N4MMQ6/"
      },
      {
        "message_id": "4E7C72B8.5000004@getdesigned.at",
        "subject": "Re: [boost] Overload resolution speed",
        "content": "On 22.09.2011 17:57, Dave Abrahams wrote:\n> on Thu Sep 22 2011, Sebastian Redl<sebastian.redl-AT-getdesigned.at>  wrote:\n>\n>> Overload resolution is supposed to be linear in the number of\n>> overloads.\n> According to whom?\nThe C++ standard has a footnote that outlines a linear algorithm for \noverload resolution.\nClang follows this algorithm, and I suspect pretty much every other \ncompiler does as well.\nTherefore, if resolution is superlinear, it's a bug.\n>> In general, all algorithms in a compiler should be linear, or worst\n>> case n*log(n). Any quadratic or worse algorithm is pretty much a bug.\n> I'd like to think so, too, but I'm not sure all implementors would agree\n> with you.\nI can't speak for any other compilers, but I'm pretty sure Ted and Doug \nwould agree with me about the main compilation pass of Clang.\nWe make exceptions for emitting errors, and of course for the static \nanalyzer, whose pass-sensitive checks are inherently exponential.\n\nSebastian\n",
        "thread_url": "http://lists.boost.org/archives/api/list/boost@lists.boost.org/thread/6UL7DQGCTGUYBLD4UCT2ZUKURBCJHHPT/",
        "parent": "http://lists.boost.org/archives/api/list/boost@lists.boost.org/email/N5RTBSLST5A73IBYPYC2F2JFGNRG5CUN/",
        "children": [
          "http://lists.boost.org/archives/api/list/boost@lists.boost.org/email/RI27TD5AAC7EQFEGTECQOEQ2LKPHMPS4/"
        ],
        "sender_address": "sebastian.redl@getdesigned.at",
        "from": "Sebastian Redl <sebastian.redl@getdesigned.at>",
        "date": "Fri, 23 Sep 2011 13:51:20 +0200",
        "to": "boost@lists.boost.org",
        "cc": "",
        "reply_to": "",
        "url": "http://lists.boost.org/archives/api/list/boost@lists.boost.org/email/IEPHFDKU6666ZJYBLKTMRCPJWFHIOTXE/"
      }
    ],
    "message_count": 2
  }
}
```

### Type 2: Email Messages Request
Request to get specific email messages.

**Endpoint:** `/api/messages/thread/new`

**Request Example:**
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "requestId": "req_emails_67890",
  "type": "NEW EMAILS",
  "data": {
    "messages": [
      {
        "message_id": "j5hkfg$msc$1@dough.gmane.org",
        "subject": "Re: [boost] Overload resolution speed",
        "content": "On 22/09/2011 00:42, John Bytheway wrote:\n\n> What about using functors; so the call would be something like\n>\n> dispatching<f0_>()(h49_<int>(), h49_<int>(), h49_<int>(), h49_<int>(),\n> h49_<int>());\n>\n> with the f parameter choosing a template specialization, and then the\n> other parameters picking an overload of operator() ion it?\n\nThe problem is that you cannot overload dispatching<f0_>::operator() \nonce the class has been defined.\n\n\n",
        "thread_url": "http://lists.boost.org/archives/api/list/boost@lists.boost.org/thread/6UL7DQGCTGUYBLD4UCT2ZUKURBCJHHPT/",
        "parent": "http://lists.boost.org/archives/api/list/boost@lists.boost.org/email/6NPNBS36CWRJCGAWBRDC4YZJSV2HFG4B/",
        "children": [],
        "sender_address": "mathias.gaunard@ens-lyon.org",
        "from": "Mathias Gaunard <mathias.gaunard@ens-lyon.org>",
        "date": "Fri, 23 Sep 2011 11:45:19 +0200",
        "to": "boost@lists.boost.org",
        "cc": "",
        "reply_to": "",
        "url": "http://lists.boost.org/archives/api/list/boost@lists.boost.org/email/YINRZWKYACLGPDC7P67X6NZ4Y6N4MMQ6/"
      },
      {
        "message_id": "4E7C72B8.5000004@getdesigned.at",
        "subject": "Re: [boost] Overload resolution speed",
        "content": "On 22.09.2011 17:57, Dave Abrahams wrote:\n> on Thu Sep 22 2011, Sebastian Redl<sebastian.redl-AT-getdesigned.at>  wrote:\n>\n>> Overload resolution is supposed to be linear in the number of\n>> overloads.\n> According to whom?\nThe C++ standard has a footnote that outlines a linear algorithm for \noverload resolution.\nClang follows this algorithm, and I suspect pretty much every other \ncompiler does as well.\nTherefore, if resolution is superlinear, it's a bug.\n>> In general, all algorithms in a compiler should be linear, or worst\n>> case n*log(n). Any quadratic or worse algorithm is pretty much a bug.\n> I'd like to think so, too, but I'm not sure all implementors would agree\n> with you.\nI can't speak for any other compilers, but I'm pretty sure Ted and Doug \nwould agree with me about the main compilation pass of Clang.\nWe make exceptions for emitting errors, and of course for the static \nanalyzer, whose pass-sensitive checks are inherently exponential.\n\nSebastian\n",
        "thread_url": "http://lists.boost.org/archives/api/list/boost@lists.boost.org/thread/6UL7DQGCTGUYBLD4UCT2ZUKURBCJHHPT/",
        "parent": "http://lists.boost.org/archives/api/list/boost@lists.boost.org/email/N5RTBSLST5A73IBYPYC2F2JFGNRG5CUN/",
        "children": [
          "http://lists.boost.org/archives/api/list/boost@lists.boost.org/email/RI27TD5AAC7EQFEGTECQOEQ2LKPHMPS4/"
        ],
        "sender_address": "sebastian.redl@getdesigned.at",
        "from": "Sebastian Redl <sebastian.redl@getdesigned.at>",
        "date": "Fri, 23 Sep 2011 13:51:20 +0200",
        "to": "boost@lists.boost.org",
        "cc": "",
        "reply_to": "",
        "url": "http://lists.boost.org/archives/api/list/boost@lists.boost.org/email/IEPHFDKU6666ZJYBLKTMRCPJWFHIOTXE/"
      }
    ],
    "message_count": 2
  }
}
```

## Response Types

### 1. Success Responses

#### Resource Created Successfully
**Response Example:**
```json
{
  "success": true,
  "message": "User created successfully",
  "data": {
    "id": 123,
    "name": "John Doe",
    "email": "john@example.com",
    "createdAt": "2024-01-15T10:30:00Z",
    "status": "active"
  }
}
```

#### Resource Updated Successfully
**Response Example:**
```json
{
  "success": true,
  "message": "User updated successfully",
  "data": {
    "id": 123,
    "name": "John Doe",
    "email": "john@example.com",
    "updatedAt": "2024-01-15T10:30:00Z"
  }
}
```

#### Request Accepted for Processing
**Response Example:**
```json
{
  "success": true,
  "message": "Request accepted for processing",
  "data": {
    "jobId": "job_12345",
    "estimatedCompletionTime": "2024-01-15T10:35:00Z"
  }
}
```

### 2. Error Responses

#### Validation Error (400)
Invalid request data or malformed request.

**Response Example:**
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": [
      {
        "field": "email",
        "message": "Invalid email format",
        "code": "INVALID_EMAIL_FORMAT"
      },
      {
        "field": "age",
        "message": "Age must be a positive number",
        "code": "INVALID_AGE_RANGE"
      }
    ]
  }
}
```

#### Required Field Missing (400)
**Response Example:**
```json
{
  "success": false,
  "error": {
    "code": "MISSING_REQUIRED_FIELD",
    "message": "Required field is missing",
    "details": [
      {
        "field": "name",
        "message": "Name is required",
        "code": "FIELD_REQUIRED"
      }
    ]
  }
}
```

#### Invalid Data Type (400)
**Response Example:**
```json
{
  "success": false,
  "error": {
    "code": "INVALID_DATA_TYPE",
    "message": "Invalid data type provided",
    "details": [
      {
        "field": "age",
        "message": "Age must be a number",
        "code": "EXPECTED_NUMBER"
      }
    ]
  }
}
```

#### Authentication Error (401)
Authentication required or invalid credentials.

**Response Example:**
```json
{
  "success": false,
  "error": {
    "code": "AUTHENTICATION_FAILED",
    "message": "Authentication required",
    "details": "Invalid or expired token"
  }
}
```

#### Invalid Credentials (401)
**Response Example:**
```json
{
  "success": false,
  "error": {
    "code": "INVALID_CREDENTIALS",
    "message": "Invalid username or password",
    "details": "The provided credentials are incorrect"
  }
}
```

#### Token Expired (401)
**Response Example:**
```json
{
  "success": false,
  "error": {
    "code": "TOKEN_EXPIRED",
    "message": "Authentication token has expired",
    "details": "Please login again to get a new token"
  }
}
```

#### Authorization Error (403)
Valid authentication but insufficient permissions.

**Response Example:**
```json
{
  "success": false,
  "error": {
    "code": "INSUFFICIENT_PERMISSIONS",
    "message": "Insufficient permissions",
    "details": "You don't have permission to access this resource"
  }
}
```

#### Access Denied (403)
**Response Example:**
```json
{
  "success": false,
  "error": {
    "code": "ACCESS_DENIED",
    "message": "Access denied",
    "details": "Your account does not have the required privileges"
  }
}
```

#### Resource Not Found (404)
Resource or endpoint not found.

**Response Example:**
```json
{
  "success": false,
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "Resource not found",
    "details": "User with ID 999 does not exist"
  }
}
```

#### Endpoint Not Found (404)
**Response Example:**
```json
{
  "success": false,
  "error": {
    "code": "ENDPOINT_NOT_FOUND",
    "message": "API endpoint not found",
    "details": "The requested endpoint does not exist"
  }
}
```

#### Conflict Error (409)
Resource already exists or conflicts with current state.

**Response Example:**
```json
{
  "success": false,
  "error": {
    "code": "RESOURCE_ALREADY_EXISTS",
    "message": "Resource already exists",
    "details": "User with email 'john@example.com' already exists"
  }
}
```

#### Duplicate Entry (409)
**Response Example:**
```json
{
  "success": false,
  "error": {
    "code": "DUPLICATE_ENTRY",
    "message": "Duplicate entry detected",
    "details": "A record with this information already exists"
  }
}
```



#### Request Payload Too Large (413)
Request content data is overloaded or exceeds size limits.

**Response Example:**
```json
{
  "success": false,
  "error": {
    "code": "PAYLOAD_TOO_LARGE",
    "message": "Request payload too large",
    "details": "Request size exceeds maximum allowed limit of 10MB",
    "maxSize": "10MB",
    "currentSize": "15MB"
  }
}
```

#### Request Content Overloaded (413)
**Response Example:**
```json
{
  "success": false,
  "error": {
    "code": "CONTENT_OVERLOADED",
    "message": "Request content is overloaded",
    "details": "Too many messages in request. Maximum allowed: 100 messages",
    "maxMessages": 100,
    "currentMessages": 150
  }
}
```

#### Memory Limit Exceeded (413)
**Response Example:**
```json
{
  "success": false,
  "error": {
    "code": "MEMORY_LIMIT_EXCEEDED",
    "message": "Request exceeds memory limits",
    "details": "Processing this request would exceed server memory limits",
    "suggestion": "Please reduce the number of messages or split into smaller requests"
  }
}
```

#### Business Logic Error (422)
Valid request format but semantic errors.

**Response Example:**
```json
{
  "success": false,
  "error": {
    "code": "BUSINESS_RULE_VIOLATION",
    "message": "Business rule violation",
    "details": [
      {
        "field": "password",
        "message": "Password must be at least 8 characters long",
        "code": "PASSWORD_TOO_SHORT"
      }
    ]
  }
}
```

#### Invalid Business Operation (422)
**Response Example:**
```json
{
  "success": false,
  "error": {
    "code": "INVALID_OPERATION",
    "message": "Operation not allowed",
    "details": "Cannot delete user with active orders"
  }
}
```

#### Rate Limit Exceeded (429)
Too many requests.

**Response Example:**
```json
{
  "success": false,
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Too many requests",
    "details": "Rate limit of 100 requests per hour exceeded",
    "retryAfter": 60
  }
}
```

#### Quota Exceeded (429)
**Response Example:**
```json
{
  "success": false,
  "error": {
    "code": "QUOTA_EXCEEDED",
    "message": "API quota exceeded",
    "details": "You have exceeded your monthly API quota"
  }
}
```

#### Server Error (500)
Server-side error.

**Response Example:**
```json
{
  "success": false,
  "error": {
    "code": "INTERNAL_SERVER_ERROR",
    "message": "Internal server error",
    "details": "An unexpected error occurred. Please try again later."
  }
}
```

#### Database Error (500)
**Response Example:**
```json
{
  "success": false,
  "error": {
    "code": "DATABASE_ERROR",
    "message": "Database operation failed",
    "details": "Unable to connect to the database"
  }
}
```

#### Service Unavailable (503)
**Response Example:**
```json
{
  "success": false,
  "error": {
    "code": "SERVICE_UNAVAILABLE",
    "message": "Service temporarily unavailable",
    "details": "The service is currently under maintenance"
  }
}
```