from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import threading
import queue
import wavio
import sounddevice as sd
import assemblyai as aai
import uvicorn
from pydantic import BaseModel
import time
from typing import List, Dict, Optional
import numpy as np
from fastapi.responses import StreamingResponse
import asyncio
import json
import google.generativeai as genai
from dotenv import load_dotenv

# === Config ===
API_KEY = "b6df08faee24484382a769da39ac8e29"
GOOGLE_API_KEY = "AIzaSyAqpitd8mjXH7gvRMAC7bA5iIXFCi7bvaw"
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SECONDS = 7
SILENCE_THRESHOLD = 0.01  # Threshold for silence detection
SILENCE_BLOCKS_TO_STOP = 10  # Number of consecutive silent blocks before stopping

aai.settings.api_key = API_KEY
transcriber = aai.Transcriber()

# Initialize the model
load_dotenv()
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

app = FastAPI(title="Audio Transcription API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranscriptionResponse(BaseModel):
    transcript: str

class TranscriptionsResponse(BaseModel):
    transcripts: List[str]

class SessionResponse(BaseModel):
    session_id: str
    status: str

class Session:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.is_recording = True
        self.audio_queue = queue.Queue()
        self.recording_thread = None
        self.transcription_thread = None
        # Ensure each session has its own isolated transcription history
        self.transcription_history = []
        self.silent_blocks_count = 0
        self.last_activity_time = time.time()
        # Track the last processed transcription to avoid duplicates within this session
        self.last_processed_transcription = None
        # Flag to track if this session is being cleaned up
        self.is_cleanup_in_progress = False
        # Event to signal threads to exit
        self.cleanup_event = threading.Event()

# Store for all active sessions
active_sessions: Dict[str, Session] = {}

def generate_ai_explanation(text: str, is_brief: bool) -> str:
    """Generate AI explanation using Google's Gemini 2.0 Flash model."""
    try:
        if is_brief:
            prompt = f"""You are a helpful AI assistant. Provide a very brief explanation (2-3 sentences) of the following text. 
            Focus on the main points and keep it concise:
            {text}"""
        else:
            prompt = f"""You are a helpful AI assistant. Provide a detailed explanation of the following text. 
            Include key points, examples, and context. Make it comprehensive and educational:
            {text}"""
        
        response = model.generate_content(prompt)
        if response and response.text:
            return response.text
        else:
            print("No response text received from the model")
            return "Error generating explanation"
    except Exception as e:
        print(f"Error generating AI explanation: {str(e)}")
        return f"Error generating explanation: {str(e)}"

def record_worker(session: Session):
    """Continuously read audio blocks from a persistent stream for a specific session."""
    print(f"Starting record_worker for session {session.session_id}")
    
    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            blocksize=BLOCK_SECONDS * SAMPLE_RATE,
            latency='low'
        )
        
        print(f"Audio stream created successfully for session {session.session_id}")
        
        with stream:
            while session.is_recording and not session.cleanup_event.is_set():
                try:
                    data, _ = stream.read(BLOCK_SECONDS * SAMPLE_RATE)
                    
                    # Calculate the RMS (Root Mean Square) of the audio block
                    rms = np.sqrt(np.mean(np.square(data)))
                    
                    # Check if this is a silent block
                    if rms <= SILENCE_THRESHOLD:
                        session.silent_blocks_count += 1
                        print(f"Session {session.session_id}: Silence detected, skipping block (count: {session.silent_blocks_count}/{SILENCE_BLOCKS_TO_STOP})")
                        
                        # If we've reached the threshold of silent blocks, auto-stop
                        if session.silent_blocks_count >= SILENCE_BLOCKS_TO_STOP:
                            print(f"Session {session.session_id}: Detected {SILENCE_BLOCKS_TO_STOP} consecutive silent blocks, stopping recording")
                            session.is_recording = False
                            break
                    else:
                        # Reset the counter if we get a non-silent block
                        session.silent_blocks_count = 0
                        session.last_activity_time = time.time()
                        session.audio_queue.put(data.copy())
                        print(f"Session {session.session_id}: Audio block recorded and queued")
                        
                except Exception as e:
                    print(f"Session {session.session_id}: Error reading audio block: {str(e)}")
                    break
                    
    except Exception as e:
        print(f"Session {session.session_id}: Error in record_worker: {str(e)}")
        session.is_recording = False
    finally:
        print(f"Session {session.session_id}: record_worker finished")
        # Signal cleanup through normal channels
        schedule_cleanup(session.session_id)

def transcribe_worker(session: Session):
    """Pull WAV blocks off the queue, save, transcribe, and print for a specific session."""
    
    print(f"Starting transcribe_worker for session {session.session_id}")
    
    while (session.is_recording or not session.audio_queue.empty()) and not session.cleanup_event.is_set():
        try:
            audio = session.audio_queue.get(timeout=1.0)
            fname = f"chunk_{session.session_id}_{uuid.uuid4().hex[:8]}.wav"
            wavio.write(fname, audio, SAMPLE_RATE, sampwidth=2)

            # Transcribe audio
            result = transcriber.transcribe(fname)
            if result.status == aai.TranscriptStatus.completed and result.text:
                # Only store if it's different from the last transcription for this specific session
                if result.text != session.last_processed_transcription:
                    session.transcription_history.append(result.text)
                    session.last_processed_transcription = result.text
                    print(f"Session {session.session_id}: Transcribed: {result.text}")
            elif result.error:
                print(f"Session {session.session_id}: Transcription Error: {result.error}")

            # Clean up
            os.remove(fname)
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Session {session.session_id}: Error in transcription worker: {e}")
    
    print(f"Session {session.session_id}: Transcription worker finished, cleaning up session")
    # Use a synchronous method to schedule cleanup
    schedule_cleanup(session.session_id)

def schedule_cleanup(session_id: str):
    """Schedule cleanup for a session using background tasks at the next API call"""
    if session_id in active_sessions:
        session = active_sessions[session_id]
        if not session.is_cleanup_in_progress:
            session.is_cleanup_in_progress = True
            session.is_recording = False
            # Signal all threads to stop
            session.cleanup_event.set()
            print(f"Cleanup scheduled for session {session_id}")

async def async_cleanup_session(session_id: str, background_tasks: BackgroundTasks):
    """Asynchronous version of cleanup_session that properly runs in the FastAPI context"""
    if session_id in active_sessions:
        print(f"Async cleanup started for session {session_id}")
        session = active_sessions[session_id]
        
        # Avoid multiple cleanup attempts
        if session.is_cleanup_in_progress:
            print(f"Cleanup already in progress for session {session_id}")
            return
        
        session.is_cleanup_in_progress = True
        session.is_recording = False
        # Signal all threads to stop
        session.cleanup_event.set()
        
        # Use background tasks to safely handle thread cleanup
        background_tasks.add_task(cleanup_threads, session_id)
        
        print(f"Session {session_id} cleanup scheduled")

def cleanup_threads(session_id: str):
    """Clean up threads for a session that's no longer active."""
    if session_id in active_sessions:
        session = active_sessions[session_id]
        
        # Wait for recording thread to finish if it exists and is running
        if session.recording_thread and session.recording_thread.is_alive():
            try:
                current_thread = threading.current_thread()
                if current_thread != session.recording_thread:
                    session.recording_thread.join(timeout=2.0)
                else:
                    print(f"Skipping join for recording thread in session {session_id} as it's the current thread")
            except RuntimeError as e:
                print(f"Error joining recording thread for session {session_id}: {e}")
        
        # Wait for transcription thread to finish if it exists and is running
        if session.transcription_thread and session.transcription_thread.is_alive():
            try:
                current_thread = threading.current_thread()
                if current_thread != session.transcription_thread:
                    session.transcription_thread.join(timeout=5.0)
                else:
                    print(f"Skipping join for transcription thread in session {session_id} as it's the current thread")
            except RuntimeError as e:
                print(f"Error joining transcription thread for session {session_id}: {e}")
        
        # Remove the session from active sessions
        if session_id in active_sessions:
            del active_sessions[session_id]
            print(f"Session {session_id} cleaned up and removed from active sessions")

def get_session(session_id: str) -> Optional[Session]:
    """Get a session by ID or return None if it doesn't exist."""
    return active_sessions.get(session_id)

def validate_session(session_id: str, background_tasks: BackgroundTasks):
    """Dependency to validate session exists."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return session

@app.post("/start", response_model=SessionResponse)
async def start_recording():
    """Start a new recording session and return the session ID."""
    # Create a unique session ID
    session_id = str(uuid.uuid4())
    
    try:
        # Create a new session
        session = Session(session_id)
        active_sessions[session_id] = session
        
        # Start the worker threads
        session.recording_thread = threading.Thread(
            target=record_worker,
            args=(session,),
            daemon=True
        )
        
        session.transcription_thread = threading.Thread(
            target=transcribe_worker,
            args=(session,),
            daemon=True
        )
        
        session.recording_thread.start()
        session.transcription_thread.start()
        
        print(f"Recording started successfully for session {session_id}")
        
        return {"session_id": session_id, "status": "started"}
        
    except Exception as e:
        print(f"Error starting recording for session {session_id}: {str(e)}")
        if session_id in active_sessions:
            del active_sessions[session_id]
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sessions/{session_id}/stop")
async def stop_recording(
    session_id: str,
    background_tasks: BackgroundTasks
):
    """Stop a specific recording session."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
    if not session.is_recording:
        return {"session_id": session_id, "status": "already stopped"}
    
    # Set the flag to stop recording
    session.is_recording = False
    session.cleanup_event.set()
    
    # Schedule cleanup in the background
    await async_cleanup_session(session_id, background_tasks)
    
    return {"session_id": session_id, "status": "stopping"}

@app.get("/sessions/{session_id}/transcripts", response_model=TranscriptionsResponse)
async def get_transcriptions(
    session_id: str,
    background_tasks: BackgroundTasks
):
    """Get all transcriptions from a specific session."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    # Return ONLY the transcriptions for the requested session
    return {"transcripts": session.transcription_history}

@app.get("/sessions/{session_id}/transcript", response_model=TranscriptionResponse)
async def get_latest_transcription(
    session_id: str,
    background_tasks: BackgroundTasks
):
    """Get the most recent transcription from a specific session."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    if not session.transcription_history:
        raise HTTPException(status_code=404, detail="No transcriptions available for this session")
    
    # Return ONLY the latest transcription from the requested session
    return {"transcript": session.transcription_history[-1]}

@app.delete("/sessions/{session_id}/transcripts")
async def clear_transcriptions(
    session_id: str,
    background_tasks: BackgroundTasks
):
    """Clear all transcriptions from a specific session."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    session.transcription_history.clear()
    session.last_processed_transcription = None
    return {"session_id": session_id, "status": "cleared"}

@app.get("/sessions")
async def list_sessions():
    """List all active sessions."""
    return {
        "active_sessions": [
            {
                "session_id": session_id,
                "is_recording": session.is_recording,
                "queue_size": session.audio_queue.qsize(),
                "transcripts_count": len(session.transcription_history),
                "silent_blocks_count": session.silent_blocks_count
            }
            for session_id, session in active_sessions.items()
        ]
    }

@app.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    background_tasks: BackgroundTasks
):
    """Force delete a session."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    # Schedule cleanup in the background
    await async_cleanup_session(session_id, background_tasks)
    return {"session_id": session_id, "status": "deleted"}

@app.get("/sessions/{session_id}/status")
async def get_session_status(
    session_id: str,
    background_tasks: BackgroundTasks
):
    """Get the current status of a specific session."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    return {
        "session_id": session.session_id,
        "recording": session.is_recording,
        "queue_size": session.audio_queue.qsize(),
        "transcripts_count": len(session.transcription_history),
        "silent_blocks_count": session.silent_blocks_count,
        "last_activity": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(session.last_activity_time))
    }

@app.get("/sessions/{session_id}/stream_explanation")
async def stream_explanation(
    session_id: str,
    background_tasks: BackgroundTasks,
    explanation_type: str = "brief"
):
    """Stream AI explanation for transcriptions from a specific session."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
    try:
        # Get transcriptions from the SPECIFIC session only
        transcript_list = session.transcription_history
        
        async def generate_explanation():
            if transcript_list:
                combined_text = " ".join(transcript_list)
                is_brief = explanation_type == "brief"
                
                # Generate the explanation using AI
                explanation = generate_ai_explanation(combined_text, is_brief)
                
                # Stream the explanation word by word
                for word in explanation.split():
                    yield f"data: {json.dumps({'text': word + ' '})}\n\n"
                    await asyncio.sleep(0.1)
            
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate_explanation(),
            media_type="text/event-stream"
        )
    except Exception as e:
        print(f"Error in stream_explanation for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task to periodically clean up inactive sessions
@app.on_event("startup")
async def start_cleanup_task():
    asyncio.create_task(cleanup_inactive_sessions())

async def cleanup_inactive_sessions():
    """Background task to clean up sessions that have been inactive for too long."""
    INACTIVE_SESSION_TIMEOUT = 300  # 5 minutes in seconds
    
    while True:
        try:
            # Check each session for inactivity
            current_time = time.time()
            sessions_to_cleanup = []
            
            for session_id, session in list(active_sessions.items()):
                if session.is_recording and current_time - session.last_activity_time > INACTIVE_SESSION_TIMEOUT:
                    print(f"Session {session_id} inactive for more than {INACTIVE_SESSION_TIMEOUT} seconds, marking for cleanup")
                    sessions_to_cleanup.append(session_id)
            
            # Clean up inactive sessions
            for session_id in sessions_to_cleanup:
                if session_id in active_sessions:
                    print(f"Auto-stopping inactive session {session_id}")
                    session = active_sessions[session_id]
                    session.is_recording = False
                    session.cleanup_event.set()
                    
                    # Clean up the session directly (in this async context)
                    if session_id in active_sessions:
                        del active_sessions[session_id]
                        print(f"Session {session_id} removed during inactive cleanup")
            
            # Sleep for a while before the next check
            await asyncio.sleep(60)  # Check every minute
            
        except Exception as e:
            print(f"Error in cleanup_inactive_sessions: {str(e)}")
            await asyncio.sleep(60)  # If error, still wait before retrying

