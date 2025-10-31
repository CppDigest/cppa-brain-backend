# Message Management API Guide

## Overview

This guide documents the RESTful API endpoints for managing messages within the RAG (Retrieval-Augmented Generation) pipeline. The API provides comprehensive functionality for ingesting, updating, and deleting messages from multiple data sources including mailing lists and Slack channels. All endpoints accept JSON-formatted request bodies and return structured responses following standard HTTP status conventions.

## 1. Message Ingestion Endpoint

### Overview

This endpoint accepts structured message data for processing through the RAG (Retrieval-Augmented Generation) pipeline. The system supports multiple data sources including mailing lists and Slack channels.

### Endpoint Details

**URL:** `/maillist/messages/new`
**Method:** `POST`  
**Content-Type:** `application/json`

---

### Request Schema

```python
{
    "timestamp": str,
    "requestId": str,
    "type": str,
    "data": {
        "messages": [
            {
                "message_id": str,
                "subject": Optional[str],
                "content": str,
                "thread_url": str,
                "parent": Optional[str],
                "children": Optional[List[str]],
                "sender_address": str,
                "from": str,
                "date": str,
                "to": Optional[str],
                "cc": Optional[str],
                "reply_to": Optional[str],
                "url": str
            }
        ],
        "message_count": int
    }
}
```

---

### Field Specifications

| Field Name             | Type            | Description                                                                                                                                    |
| ---------------------- | --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **timestamp**          | `str`           | Request timestamp in ISO 8601 format indicating when the request was created or sent to the RAG pipeline.                                      |
| **requestId**          | `str`           | Unique identifier for the data transmission request. Must include a source prefix (`slack@...` or `maillist@...`) to indicate the data origin. |
| **type**               | `str`           | Category or source type of the data being transmitted (e.g., "NEW EMAILS", "SLACK_MESSAGES").                                                  |
| **data**               | `dict`          | Primary payload container for message array and associated metadata.                                                                           |
| ┗ **messages**         | `List`          | Collection of message objects containing detailed conversation information.                                                                    |
| ┃ ┗ **message_id**     | `str`           | Unique message identifier. For Slack messages, must include the `@@Slack@@` prefix.                                                            |
| ┃ ┗ **subject**        | `Optional[str]` | (Optional) Message subject or title, typically used for email communications.                                                                  |
| ┃ ┗ **content**        | `str`           | Primary message body used by the RAG pipeline for text retrieval and generation operations.                                                    |
| ┃ ┗ **thread_url**     | `str`           | URL reference to the conversation thread. For root messages, this value equals the `url` field.                                                |
| ┃ ┗ **parent**         | `Optional[str]` | (Optional) Message ID of the parent message in threaded conversations. Should be null or undefined for root messages.                          |
| ┃ ┗ **children**       | `List[str]`     | Collection of message IDs representing direct replies or child messages. Returns empty array if no replies exist.                              |
| ┃ ┗ **sender_address** | `str`           | Email address or unique identifier of the message sender.                                                                                      |
| ┃ ┗ **from**           | `str`           | Display name and identifier of the message sender (e.g., "John Doe &lt;john@example.com&gt;").                                                 |
| ┃ ┗ **date**           | `str`           | Message creation timestamp formatted as ISO 8601 string.                                                                                       |
| ┃ ┗ **to**             | `Optional[str]` | (Optional) Comma-separated list of primary recipients. Applicable for email or direct messaging.                                               |
| ┃ ┗ **cc**             | `Optional[str]` | (Optional) Comma-separated list of carbon copy recipients.                                                                                     |
| ┃ ┗ **reply_to**       | `Optional[str]` | (Optional) Reply-to address or message identifier for routing responses.                                                                       |
| ┃ ┗ **url**            | `str`           | Direct URL reference to the individual message. Must equal `thread_url` for root messages.                                                     |
| ┗ **message_count**    | `int`           | Total count of messages included in the current request payload.                                                                               |

---

## 2. Message Update Endpoint

### Overview

This endpoint allows updating existing message properties in the mail hierarchical graph database. Only the fields provided in the request will be updated, while others remain unchanged.

### Endpoint Details

**URL:** `/maillist/message/update`  
**Method:** `PUT`  
**Content-Type:** `application/json`

---

### Request Schema

```python
{
    "message_id": str,
    "subject": Optional[str],
    "content": Optional[str],
    "sender_address": Optional[str],
    "from": Optional[str],
    "date": Optional[str],
    "to": Optional[str],
    "cc": Optional[str],
    "reply_to": Optional[str],
    "url": Optional[str]
}
```

---

### Field Specifications

| Field Name         | Type            | Description                                                                  |
| ------------------ | --------------- | ---------------------------------------------------------------------------- |
| **message_id**     | `str`           | Unique identifier of the message to update. This field is required.          |
| **subject**        | `Optional[str]` | (Optional) Updated message subject or title.                                 |
| **content**        | `Optional[str]` | (Optional) Updated message body content.                                     |
| **sender_address** | `Optional[str]` | (Optional) Updated email address or unique identifier of the message sender. |
| **from**           | `Optional[str]` | (Optional) Updated display name and identifier of the message sender.        |
| **date**           | `Optional[str]` | (Optional) Updated message creation timestamp.                               |
| **to**             | `Optional[str]` | (Optional) Updated comma-separated list of primary recipients.               |
| **cc**             | `Optional[str]` | (Optional) Updated comma-separated list of carbon copy recipients.           |
| **reply_to**       | `Optional[str]` | (Optional) Updated reply-to address or message identifier.                   |
| **url**            | `Optional[str]` | (Optional) Updated direct URL reference to the message.                      |

---

## 3. Message Deletion Endpoint

### Overview

This endpoint removes a message from the mail hierarchical graph database. The deletion includes removing the message node, its relationships, embeddings, and summaries from the system.

### Endpoint Details

**URL:** `/maillist/message/delete`  
**Method:** `DELETE`  
**Content-Type:** `application/json`

---

### Request Schema

```python
{
    "message_id": str
}
```

---

### Field Specifications

| Field Name     | Type  | Description                                                         |
| -------------- | ----- | ------------------------------------------------------------------- |
| **message_id** | `str` | Unique identifier of the message to delete. This field is required. |

---

## Response Types

### 1. Success Responses

#### Resource Created Successfully (200)

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

#### Resource Updated Successfully (200)

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
