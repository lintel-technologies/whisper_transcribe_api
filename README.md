# Whisper Transcribe API

# Whisper API Documentation

The **Whisper API** allows you to interact with audio transcription services, check transcription progress, and query the transcription queue.

### Authentication
All API requests require a **Bearer Token** for authentication set in config.json. 

Include the following authorization header in your requests:

Authorization: Bearer securepassword


## Endpoints

### 1. Transcribe Audio / Video

- **URL**: `/transcribe`
- **Method**: `POST`
- **Authentication**: Bearer Token
- **Body**:
  - `audio` (file): The audio or video file to be transcribed.

#### Example Request:
```bash
curl -X POST http://url:5000/transcribe \
  -H "Authorization: Bearer securepassword" \
  -F "audio=@/path/to/audio/file.mp3"
```


### 2. Check Transcription Progress

- **URL**: `/transcribe/status/{id}`
- **Method**: `GET`
- **Authentication**: Bearer Token
- **Path Parameter**:
  - `{id}`: The ID of the transcription task.

#### Example Request:
```bash
curl -X GET http://url:5000/transcribe/status/1 \
  -H "Authorization: Bearer securepassword"
```

### 3. View Transcription Queue

- **URL**: `/queue`
- **Method**: `GET`
- **Authentication**: Bearer Token

#### Example Request:
```bash
curl -X GET http://url:5000/queue \
  -H "Authorization: Bearer securepassword"
```

### 4. Check API Health

- **URL**: `/queue`
- **Method**: `GET`
- **Authentication**: Bearer Token

#### Example Request:
```bash
curl -X GET http://url:5000/queue \
  -H "Authorization: Bearer securepassword"
```

### HELP NEEDED.

This was a quick implementation, completed in a few hours, using Whisper.cpp, ffmpeg, Python, and Docker to create a transcription API service that transcribes audio from both video and audio files.

This worked great for some training videos that I transcribed, but on the other hand, transcription from phone recordings wasn't as accurate. I believe the model needs fine-tuning.

Please help me improve this project to streamline its implementation.

Thank you in advance.

### Links

- [Whisper.cpp on GitHub](https://github.com/ggerganov/whisper.cpp)
- [FFmpeg Official Website](https://www.ffmpeg.org/)
- [Docker Official Website](https://www.docker.com/)
- [Fine-Tune Whisper on Hugging Face](https://huggingface.co/blog/fine-tune-whisper)
