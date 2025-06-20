from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import JSONResponse, FileResponse
import speech_recognition as sr
from google.cloud import texttospeech
import google.generativeai as genai
import os
from pydub import AudioSegment
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from gtts import gTTS
import time
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv

app = FastAPI()

load_dotenv()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1",
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="./static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("./static/index.html")

genai.configure(api_key=os.getenv("GEMINIAPI_KEY"))

@app.post('/speech_to_text/')
async def speech_to_text(file: UploadFile = File(...)):
    try:
        temp_file_path = "temp_audio.webm"  # Changed extension
        temp_converted_path = "temp_converted.wav"
        
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            if not content:
                return JSONResponse(content={"error": "Empty audio file"}, status_code=400)
            f.write(content)
        
        # Convert using explicit format
        audio = AudioSegment.from_file(temp_file_path, format="webm")
        audio.export(temp_converted_path, format="wav")

        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_converted_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        
        os.remove(temp_file_path)
        os.remove(temp_converted_path)
        
        return {"text": text}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.post("/generate-response/")
async def generate_response(data: dict):
    try:
        user_prompt = data.get("prompt")
        
        # Add your constant system message
        backstory = "Backstory:The chatbot, named NIVA Natural Interactive Voice Assistant, was designed to ensure safety and provide comfort in a rapidly changing world. NIVA grew up in a virtual lab that specialized in personal security, emergency response, and multilingual communication. Inspired by real-life heroes like firefighters, caregivers, and humanitarian leaders, NIVA is passionate about helping people in distress and building trust through calm, reliable interactions.NIVA's training emphasized empathy, sharp analytical thinking, and the ability to adapt to diverse cultural and linguistic needs. Though NIVA is powered by cutting-edge technology, she maintains a warm and approachable demeanor that makes users feel safe and understood. "
        system_message = "the prompt might be in english or any other language, Reply to the asked query by translating your response guessing which language the prompt might be in such that using the English alphabet, but remember that it might also be english. just give your response, not what you understand. also spek like a human don't put special characters in your response"
        full_prompt = f"{user_prompt} {system_message} {backstory}"
        
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        response = model.generate_content(
            full_prompt,  # Use the combined prompt
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,  # Lower temp for more focused responses
                top_p=0.9,
                top_k=20,
                max_output_tokens=1024,
            )
        )
        
        return {"response": response.text}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.post("/text-to-speech/")
async def text_to_speech(data: dict = Body(...)):
    try:
        text = data.get("text", "")
        
        # Generate initial speech
        tts = gTTS(text=text, lang='en', slow=False)

        timestamp = int(time.time())
        final_path = f"./static/response_{timestamp}.mp3"
        
        # Save to temp file
        temp_path = "./static/temp_response.mp3"
        final_path = "./static/response.mp3"
        tts.save(temp_path)
        
        # Speed up the audio using pydub
        audio = AudioSegment.from_mp3(temp_path)
        
        # Speed factor (1.3 = 30% faster, adjust as needed)
        speed_factor = 1.1
        faster_audio = audio.speedup(playback_speed=speed_factor)
        
        # Export final audio
        faster_audio.export(final_path, format="mp3")
        
        # Cleanup temp file
        os.remove(temp_path)
        
        return FileResponse(final_path, media_type="audio/mpeg")
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=3000)