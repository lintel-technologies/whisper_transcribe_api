import os
import json
import subprocess
from datetime import datetime
from flask import Flask, jsonify, request
import threading
import time
import uuid
import hashlib
import logging
from werkzeug.utils import secure_filename
from redis import Redis
import pickle

# Initialize Flask
app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Redis with retries
def get_redis_connection(max_retries=5, retry_delay=1):
    for attempt in range(max_retries):
        try:
            redis_client = Redis(host='redis', port=6379, db=0, decode_responses=False)
            redis_client.ping()
            logger.info("Successfully connected to Redis")
            return redis_client
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed to connect to Redis after {max_retries} attempts: {e}")
                raise
            logger.warning(f"Redis connection attempt {attempt + 1} failed, retrying...")
            time.sleep(retry_delay)

redis_client = get_redis_connection()

# Load configuration with error handling
try:
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_path) as f:
        config = json.load(f)
    logger.info("Configuration loaded successfully")
except Exception as e:
    logger.error(f"Failed to load config.json: {e}")
    raise

# Create temp directories
os.makedirs(config['temp_paths']['input'], exist_ok=True)
os.makedirs(config['temp_paths']['output'], exist_ok=True)
logger.info("Temporary directories created/verified")

# Verify paths
if not os.path.exists(config["ffmpeg_path"]):
    logger.error(f"FFmpeg not found at {config['ffmpeg_path']}")
    raise FileNotFoundError(f"FFmpeg not found at {config['ffmpeg_path']}")

# Constants
TASK_EXPIRY = config["task_expiry"]
QUEUE_KEY = "transcription:queue"
TASK_KEY_PREFIX = "transcription:task:"
FFMPEG_PATH = config["ffmpeg_path"]
FFMPEG_OPTIONS = config["ffmpeg_options"]
WHISPER_CPP_PATH = config["whisper_cpp_path"]
WHISPER_MODEL = config["whisper_model"]
LANGUAGE = config["language"]

# Redis helper functions
def get_task_key(job_id):
    return f"{TASK_KEY_PREFIX}{job_id}"

def save_task(job_id, task_data):
    task_key = get_task_key(job_id)
    redis_client.setex(task_key, TASK_EXPIRY, pickle.dumps(task_data))

def get_task(job_id):
    task_key = get_task_key(job_id)
    task_data = redis_client.get(task_key)
    return pickle.loads(task_data) if task_data else None

def add_to_queue(job_id):
    redis_client.lpush(QUEUE_KEY, job_id)

def remove_from_queue(job_id):
    redis_client.lrem(QUEUE_KEY, 0, job_id)

def get_queue():
    return [job_id.decode() for job_id in redis_client.lrange(QUEUE_KEY, 0, -1)]

def generate_job_id():
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    unique_value = str(uuid.uuid4())
    data_to_hash = current_time + unique_value
    return hashlib.md5(data_to_hash.encode()).hexdigest()

def cleanup_files(input_file, output_file):
    try:
        if os.path.exists(input_file):
            os.remove(input_file)
        if os.path.exists(output_file):
            os.remove(output_file)
    except Exception as e:
        logger.error(f"Error cleaning up files: {e}")

def convert_to_wav(input_file, output_file):
    try:
        start_time = time.time()
        ffmpeg_command = [
            FFMPEG_PATH, '-i', input_file, 
            '-ar', str(FFMPEG_OPTIONS['sample_rate']),
            '-ac', str(FFMPEG_OPTIONS['channels']),
            '-c:a', FFMPEG_OPTIONS['audio_codec'],
            '-y',
            output_file
        ]
        result = subprocess.run(ffmpeg_command, capture_output=True, text=True, check=True)
        conversion_time = time.time() - start_time
        logger.info(f"FFmpeg conversion successful for {input_file} (took {conversion_time:.2f}s)")
        return True, conversion_time
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during FFmpeg conversion: {str(e)}\nOutput: {e.output}")
        return False, 0
        
def get_file_size(file_path):
    """Get file size in MB"""
    try:
        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)
        return round(size_mb, 2)
    except Exception as e:
        logger.error(f"Error getting file size for {file_path}: {e}")
        return 0

def transcribe_with_whisper(wav_file):
    try:
        start_time = time.time()
        whisper_command = [
            WHISPER_CPP_PATH,
            '-m', WHISPER_MODEL,
            '-f', wav_file,
            '-l', LANGUAGE,
            '-nt',
            '-ls'
        ]
        result = subprocess.run(whisper_command, capture_output=True, text=True, check=True)
        transcription_time = time.time() - start_time
        
        if result.stderr:
            logger.warning(f"Whisper warnings: {result.stderr}")
        return {
            "transcription": result.stdout.strip(),
            "transcription_time": transcription_time
        }
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during Whisper transcription: {str(e)}\nOutput: {e.output}")
        return {"error": f"Transcription failed: {str(e)}"}

def process_transcription(job_id, input_file, output_file):
    try:
        start_time = time.time()
        task_data = get_task(job_id)
        if not task_data:
            logger.error(f"Task data not found for job {job_id}")
            return

        # Record input file size
        input_size = get_file_size(input_file)
        task_data['input_file_size'] = f"{input_size} MB"
        
        # Update status to Converting
        task_data['status'] = 'Converting'
        save_task(job_id, task_data)
        logger.info(f"Starting conversion for job {job_id}")
        
        success, conversion_time = convert_to_wav(input_file, output_file)
        task_data['conversion_time'] = f"{conversion_time:.2f}s"
        
        if success:
            # Record converted WAV file size
            output_size = get_file_size(output_file)
            task_data['converted_file_size'] = f"{output_size} MB"
            
            # Update status to Transcribing
            task_data['status'] = 'Transcribing'
            save_task(job_id, task_data)
            logger.info(f"Starting transcription for job {job_id}")
            
            transcription_result = transcribe_with_whisper(output_file)
            if 'transcription' in transcription_result:
                end_time = datetime.now()
                task_data['status'] = 'Completed'
                task_data['transcription'] = transcription_result['transcription']
                task_data['transcription_time'] = f"{transcription_result['transcription_time']:.2f}s"
                task_data['start_time'] = task_data['start_time']
                task_data['end_time'] = end_time.isoformat()
                task_data['total_processing_time'] = f"{(time.time() - start_time):.2f}s"
                
                # Calculate time difference
                start_dt = datetime.fromisoformat(task_data['start_time'])
                time_diff = end_time - start_dt
                task_data['total_wall_time'] = f"{time_diff.total_seconds():.2f}s"
                
                logger.info(f"Transcription completed for job {job_id}")
            else:
                task_data['status'] = 'Failed'
                task_data['error'] = transcription_result.get('error', 'Unknown error')
                logger.error(f"Transcription failed for job {job_id}")
        else:
            task_data['status'] = 'Conversion failed'
            logger.error(f"Conversion failed for job {job_id}")

        save_task(job_id, task_data)

    except Exception as e:
        logger.error(f"Error processing transcription for job {job_id}: {str(e)}")
        task_data = get_task(job_id)
        if task_data:
            task_data['status'] = 'Failed'
            task_data['error'] = str(e)
            task_data['end_time'] = datetime.now().isoformat()
            save_task(job_id, task_data)
    finally:
        remove_from_queue(job_id)
        cleanup_files(input_file, output_file)

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if config["api_key_required"]:
        auth_header = request.headers.get('Authorization', '')
        api_key = auth_header.split(" ")[-1] if auth_header else ''
        if api_key not in config["api_keys"]:
            return jsonify({"error": "Unauthorized"}), 401

    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_extension = file.filename.rsplit('.', 1)[-1].lower()
    if file_extension not in config['allowed_formats']:
        return jsonify({"error": f"Invalid format. Allowed: {', '.join(config['allowed_formats'])}"}), 400

    file.seek(0, os.SEEK_END)
    file_size_mb = file.tell() / (1024 * 1024)
    file.seek(0)
    if file_size_mb > config["max_file_size_mb"]:
        return jsonify({"error": f"File too large. Max: {config['max_file_size_mb']} MB"}), 400

    input_filename = secure_filename(file.filename)
    input_filepath = os.path.join(config['temp_paths']['input'], f"{uuid.uuid4()}_{input_filename}")
    output_filepath = os.path.join(config['temp_paths']['output'], f"{uuid.uuid4()}.wav")

    try:
        file.save(input_filepath)
        logger.info(f"File saved: {input_filepath}")
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        return jsonify({"error": "Failed to save file"}), 500

    job_id = generate_job_id()
    task_data = {
        'status': 'Queued',
        'start_time': datetime.now().isoformat(),
        'input_file': input_filepath,
        'output_file': output_filepath,
        'original_filename': input_filename,
        'original_file_size': f"{file_size_mb:.2f} MB"
    }
    
    save_task(job_id, task_data)
    add_to_queue(job_id)
    
    threading.Thread(target=process_transcription, args=(job_id, input_filepath, output_filepath)).start()
    logger.info(f"Job {job_id} started")

    return jsonify({
        "job_id": job_id,
        "status": "Queued",
        "message": "Transcription job started"
    }), 202

@app.route('/transcribe/status/<job_id>', methods=['GET'])
def check_transcription_status(job_id):
    task_data = get_task(job_id)
    if not task_data:
        return jsonify({"error": "Job not found"}), 404
    
    response_data = task_data.copy()
    response_data.pop('input_file', None)
    response_data.pop('output_file', None)
    return jsonify(response_data)

@app.route('/queue', methods=['GET'])
def check_queue_status():
    queue_list = get_queue()
    active_tasks = sum(1 for job_id in queue_list if get_task(job_id) is not None)
    
    return jsonify({
        "jobs_in_queue": len(queue_list),
        "active_tasks": active_tasks,
        "queue": queue_list
    })

@app.route('/health', methods=['GET'])
def health_check():
    health_status = {
        "status": "healthy",
        "components": {
            "redis": {"status": "unknown"},
            "ffmpeg": {"status": "unknown"},
            "whisper": {"status": "unknown"}
        }
    }

    # Check Redis
    try:
        redis_client.ping()
        health_status["components"]["redis"] = {
            "status": "healthy",
            "message": "Connected successfully"
        }
    except Exception as e:
        health_status["components"]["redis"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "unhealthy"

    # Check FFmpeg
    try:
        result = subprocess.run(
            [FFMPEG_PATH, "-version"], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            health_status["components"]["ffmpeg"] = {
                "status": "healthy",
                "message": f"Available: {version}"
            }
        else:
            health_status["components"]["ffmpeg"] = {
                "status": "unhealthy",
                "error": "FFmpeg returned non-zero exit code"
            }
            health_status["status"] = "unhealthy"
    except Exception as e:
        health_status["components"]["ffmpeg"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "unhealthy"

    # Check Whisper CPP
    try:
        result = subprocess.run(
            [WHISPER_CPP_PATH, "--model", WHISPER_MODEL, "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            health_status["components"]["whisper"] = {
                "status": "healthy",
                "message": "Whisper CPP available and model loaded"
            }
        else:
            health_status["components"]["whisper"] = {
                "status": "unhealthy",
                "error": "Whisper CPP returned non-zero exit code"
            }
            health_status["status"] = "unhealthy"
    except Exception as e:
        health_status["components"]["whisper"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "unhealthy"

    status_code = 200 if health_status["status"] == "healthy" else 500
    return jsonify(health_status), status_code

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
