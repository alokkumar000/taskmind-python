import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()  # Safely load local environment variables

app = FastAPI()

# 1. Enable CORS for local React runtime environment iterations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Initialize the Google Gen AI Client
try:
    client = genai.Client()
except Exception as e:
    print(f"Error initializing Gemini Client. Details: {e}")
    client = None

# 3. Request Schema from Frontend
class TaskRequest(BaseModel):
    tasks: list[str]

# --- Structured Output Schema matching your exact desired structure ---
class OptimizationResponseSchema(BaseModel):
    orderedTasks: list[str] = Field(
        description="The complete list of input tasks rearranged into a high-productivity chronological sequence. Plain text strings only."
    )
    reason: str = Field(
        description="A cohesive, overarching explanation or strategic breakdown of why this specific sequence maximizes focus and efficiency with minimal distraction and cognitive load with human-like reasoning."
    )

@app.post("/api/optimize")
async def optimize_tasks(request: TaskRequest):
    if not client:
        raise HTTPException(status_code=500, detail="Gemini API Client is not configured.")
        
    # Matches your exact fallback requirement structure cleanly
    if not request.tasks:
        return {"orderedTasks": [], "reason": ""}

    formatted_tasks = "\n".join([f"- {task}" for task in request.tasks])

    # 4. Construct the prompt directing Gemini to populate both fields
    system_instruction = (
        "You are an expert productivity assistant. Analyze the incoming task array, "
        "group contexts naturally (like communication, heavy technical focus, admin work), "
        "and determine the single most efficient chronological flow to minimize exhaustion. "
        "Populate the 'orderedTasks' list field with the re-arranged plain task strings. "
        "Populate the 'reason' string field with a comprehensive, brief summary explaining the "
        "underlying productivity strategy behind this specific task order."
    )

    try:
        # 5. Request Structured Output using gemini-3.8-flash
        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=f"Optimize this task batch:\n{formatted_tasks}",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3,
                response_mime_type="application/json",
                response_schema=OptimizationResponseSchema,
            )
        )

        if response.text:
            parsed_json = json.loads(response.text)
            return parsed_json
        else:
            raise ValueError("Empty generation block returned from provider.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Structured AI Pipeline Fail: {str(e)}")
