# Multipart Upload Implementation Guide

This guide explains how to implement multipart file uploads for the AceVision Backend API when dealing with large video files (>200MB).

## Overview

The API supports two upload modes:
1. **Single Upload**: For files smaller than the configured chunk size (default: 20MB)
2. **Multipart Upload**: For files larger than or equal to the chunk size

## API Endpoints

### 1. Create Task and Initiate Upload
**Endpoint**: `POST /process_video`

**Request**:
- `name` (Form): Video name/description
- `video_file` (File): The video file (can be first chunk or full file if < 20MB)
- `total_size` (Form, optional): Total file size in bytes (required for multipart)

**Response** (Multipart Required):
```json
{
  "success": true,
  "message": "Task created. Please upload chunks using /upload_chunk/{task_id}",
  "data": {
    "process_id": "123",
    "filename": "video.mp4",
    "name": "My Video",
    "file_path": "./uploads/uuid.mp4",
    "total_size": 250000000,
    "chunk_size": 20971520,
    "requires_multipart": true
  }
}
```

**Response** (Single Upload):
```json
{
  "success": true,
  "message": "Video uploaded and queued for processing",
  "data": {
    "process_id": "123",
    "filename": "video.mp4",
    "name": "My Video",
    "file_path": "./uploads/uuid.mp4",
    "requires_multipart": false
  }
}
```

### 2. Upload Chunk
**Endpoint**: `POST /upload_chunk/{task_id}`

**Request**:
- `chunk_number` (Form): Zero-based or one-based chunk index (your choice, but be consistent)
- `chunk_data` (File): The chunk binary data
- `total_chunks` (Form, optional): Total number of chunks (for progress tracking)

**Response** (Upload In Progress):
```json
{
  "success": true,
  "message": "Chunk 0 uploaded successfully",
  "data": {
    "task_id": 123,
    "chunk_number": 0,
    "chunk_size": 20971520,
    "uploaded_size": 20971520,
    "total_size": 250000000,
    "upload_complete": false,
    "progress_percent": 8.39
  }
}
```

**Response** (Upload Complete):
```json
{
  "success": true,
  "message": "All chunks uploaded. Video processing started.",
  "data": {
    "task_id": 123,
    "uploaded_size": 250000000,
    "total_size": 250000000,
    "upload_complete": true,
    "chunks_received": 12
  }
}
```

### 3. Check Upload Progress
**Endpoint**: `GET /task_progress/{process_id}`

**Response**:
```json
{
  "success": true,
  "data": {
    "process_id": 123,
    "progress": 0.0,
    "status": "uploading",
    "description": "Uploading video: My Video",
    "total_upload_size": 250000000,
    "uploaded_size": 125000000,
    "is_uploaded_fully": false,
    "upload_progress_percent": 50.0
  }
}
```

## Implementation Flow

### Step 1: Determine Upload Mode

```javascript
const CHUNK_SIZE = 20 * 1024 * 1024; // 20MB
const file = document.getElementById('fileInput').files[0];
const fileSize = file.size;
const needsMultipart = fileSize >= CHUNK_SIZE;
```

### Step 2: Single Upload (< 20MB)

```javascript
async function uploadSingleFile(file, name) {
  const formData = new FormData();
  formData.append('name', name);
  formData.append('video_file', file);
  // total_size is optional for small files
  
  const response = await fetch('/process_video', {
    method: 'POST',
    body: formData
  });
  
  const result = await response.json();
  return result.data.process_id;
}
```

### Step 3: Multipart Upload (>= 20MB)

#### 3.1 Create Task

```javascript
async function createMultipartTask(file, name) {
  const formData = new FormData();
  formData.append('name', name);
  formData.append('total_size', file.size);
  // Optionally send first chunk or empty file
  formData.append('video_file', new Blob([]), 'placeholder');
  
  const response = await fetch('/process_video', {
    method: 'POST',
    body: formData
  });
  
  const result = await response.json();
  if (result.data.requires_multipart) {
    return result.data.process_id;
  }
  throw new Error('Expected multipart upload');
}
```

#### 3.2 Split File into Chunks

```javascript
function splitFileIntoChunks(file, chunkSize) {
  const chunks = [];
  let start = 0;
  let chunkIndex = 0;
  
  while (start < file.size) {
    const end = Math.min(start + chunkSize, file.size);
    const chunk = file.slice(start, end);
    chunks.push({
      index: chunkIndex,
      data: chunk,
      start: start,
      end: end
    });
    start = end;
    chunkIndex++;
  }
  
  return chunks;
}
```

#### 3.3 Upload Chunks (Flexible Ordering)

The API supports flexible chunk ordering, so you can upload chunks in parallel or any order:

```javascript
async function uploadChunk(taskId, chunk, chunkIndex, totalChunks) {
  const formData = new FormData();
  formData.append('chunk_number', chunkIndex);
  formData.append('chunk_data', chunk.data, `chunk_${chunkIndex}`);
  formData.append('total_chunks', totalChunks);
  
  const response = await fetch(`/upload_chunk/${taskId}`, {
    method: 'POST',
    body: formData
  });
  
  return await response.json();
}

// Sequential upload
async function uploadChunksSequential(taskId, chunks) {
  const totalChunks = chunks.length;
  
  for (const chunk of chunks) {
    const result = await uploadChunk(taskId, chunk, chunk.index, totalChunks);
    
    if (result.data.upload_complete) {
      console.log('Upload complete!');
      return result;
    }
    
    // Update progress UI
    updateProgressBar(result.data.progress_percent);
  }
}

// Parallel upload (faster, but more complex)
async function uploadChunksParallel(taskId, chunks, maxConcurrent = 3) {
  const totalChunks = chunks.length;
  const uploadPromises = [];
  
  for (let i = 0; i < chunks.length; i += maxConcurrent) {
    const batch = chunks.slice(i, i + maxConcurrent);
    
    const batchPromises = batch.map(chunk => 
      uploadChunk(taskId, chunk, chunk.index, totalChunks)
    );
    
    const results = await Promise.all(batchPromises);
    
    // Check if any upload completed
    const completed = results.find(r => r.data.upload_complete);
    if (completed) {
      return completed;
    }
    
    // Update progress from latest result
    const latestResult = results[results.length - 1];
    updateProgressBar(latestResult.data.progress_percent);
  }
}
```

### Step 4: Complete Example

```javascript
async function uploadVideo(file, name) {
  const CHUNK_SIZE = 20 * 1024 * 1024; // 20MB
  const fileSize = file.size;
  
  try {
    if (fileSize < CHUNK_SIZE) {
      // Single upload
      console.log('Using single upload mode');
      const taskId = await uploadSingleFile(file, name);
      return taskId;
    } else {
      // Multipart upload
      console.log('Using multipart upload mode');
      
      // Step 1: Create task
      const taskId = await createMultipartTask(file, name);
      console.log(`Task created: ${taskId}`);
      
      // Step 2: Split file
      const chunks = splitFileIntoChunks(file, CHUNK_SIZE);
      console.log(`Split into ${chunks.length} chunks`);
      
      // Step 3: Upload chunks
      const result = await uploadChunksSequential(taskId, chunks);
      
      if (result.data.upload_complete) {
        console.log('All chunks uploaded. Processing started.');
        return taskId;
      }
    }
  } catch (error) {
    console.error('Upload failed:', error);
    throw error;
  }
}
```

## Progress Tracking

### Polling Method

```javascript
async function pollUploadProgress(taskId, onProgress) {
  const interval = setInterval(async () => {
    const response = await fetch(`/task_progress/${taskId}`);
    const result = await response.json();
    
    if (result.success && result.data) {
      const progress = result.data;
      
      onProgress({
        uploadProgress: progress.upload_progress_percent,
        isUploaded: progress.is_uploaded_fully,
        processingProgress: progress.progress,
        status: progress.status
      });
      
      // Stop polling if upload is complete and processing started
      if (progress.is_uploaded_fully && progress.status !== 'uploading') {
        clearInterval(interval);
      }
    }
  }, 1000); // Poll every second
  
  return interval;
}

// Usage
const taskId = await uploadVideo(file, name);
const pollInterval = pollUploadProgress(taskId, (progress) => {
  console.log(`Upload: ${progress.uploadProgress}%`);
  console.log(`Processing: ${progress.processingProgress}%`);
});
```

## Error Handling

### Retry Logic for Failed Chunks

```javascript
async function uploadChunkWithRetry(taskId, chunk, chunkIndex, maxRetries = 3) {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await uploadChunk(taskId, chunk, chunkIndex);
    } catch (error) {
      if (attempt === maxRetries - 1) {
        throw new Error(`Failed to upload chunk ${chunkIndex} after ${maxRetries} attempts`);
      }
      console.warn(`Chunk ${chunkIndex} upload failed, retrying... (${attempt + 1}/${maxRetries})`);
      await new Promise(resolve => setTimeout(resolve, 1000 * (attempt + 1))); // Exponential backoff
    }
  }
}
```

### Handle Partial Uploads

If upload fails partway through, you can resume by checking which chunks are already uploaded:

```javascript
async function resumeUpload(taskId, file, name) {
  // Check current progress
  const progressResponse = await fetch(`/task_progress/${taskId}`);
  const progress = await progressResponse.json();
  
  if (progress.data.is_uploaded_fully) {
    console.log('Upload already complete');
    return;
  }
  
  // Calculate which chunks still need to be uploaded
  const CHUNK_SIZE = 20 * 1024 * 1024;
  const uploadedSize = progress.data.uploaded_size;
  const startChunk = Math.floor(uploadedSize / CHUNK_SIZE);
  
  const chunks = splitFileIntoChunks(file, CHUNK_SIZE);
  const remainingChunks = chunks.slice(startChunk);
  
  // Continue uploading from where we left off
  await uploadChunksSequential(taskId, remainingChunks);
}
```

## React Example

```jsx
import React, { useState } from 'react';

function VideoUploader() {
  const [uploadProgress, setUploadProgress] = useState(0);
  const [status, setStatus] = useState('idle');
  const [taskId, setTaskId] = useState(null);

  const handleFileUpload = async (file, name) => {
    setStatus('uploading');
    
    try {
      const CHUNK_SIZE = 20 * 1024 * 1024;
      
      if (file.size < CHUNK_SIZE) {
        // Single upload
        const formData = new FormData();
        formData.append('name', name);
        formData.append('video_file', file);
        
        const response = await fetch('/process_video', {
          method: 'POST',
          body: formData
        });
        
        const result = await response.json();
        setTaskId(result.data.process_id);
        setStatus('processing');
      } else {
        // Multipart upload
        const taskId = await createMultipartTask(file, name);
        setTaskId(taskId);
        
        const chunks = splitFileIntoChunks(file, CHUNK_SIZE);
        
        for (const chunk of chunks) {
          const result = await uploadChunk(taskId, chunk, chunk.index, chunks.length);
          setUploadProgress(result.data.progress_percent);
          
          if (result.data.upload_complete) {
            setStatus('processing');
            break;
          }
        }
      }
    } catch (error) {
      setStatus('error');
      console.error('Upload failed:', error);
    }
  };

  return (
    <div>
      <input
        type="file"
        accept="video/*"
        onChange={(e) => {
          const file = e.target.files[0];
          if (file) {
            handleFileUpload(file, file.name);
          }
        }}
      />
      {status === 'uploading' && (
        <div>
          <progress value={uploadProgress} max={100} />
          <span>{uploadProgress.toFixed(2)}%</span>
        </div>
      )}
      {taskId && <p>Task ID: {taskId}</p>}
    </div>
  );
}
```

## Best Practices

1. **Chunk Size**: Use the chunk size returned by the API (default: 20MB) to ensure consistency
2. **Error Handling**: Always implement retry logic for chunk uploads
3. **Progress Tracking**: Poll `/task_progress/{task_id}` to show both upload and processing progress
4. **Parallel Uploads**: Consider uploading chunks in parallel (3-5 concurrent) for faster uploads
5. **Resume Support**: Store task_id and implement resume functionality for failed uploads
6. **Validation**: Validate file type and size before starting upload
7. **User Feedback**: Show clear progress indicators for both upload and processing phases

## Configuration

The chunk size is configurable via the `UPLOAD_CHUNK_SIZE` environment variable (in bytes). Default is 20MB (20971520 bytes).

## Notes

- Chunks can be uploaded in any order (flexible ordering)
- The API automatically reassembles chunks in the correct order
- Upload progress is tracked in real-time
- Processing only starts after all chunks are uploaded
- Incomplete uploads are not re-queued on server restart

