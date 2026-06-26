import asyncio
import json
import os
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import google.generativeai as genai

app = FastAPI(
    title="Multi-Agent Research & Debate Engine",
    description="Asynchronous SSE streaming engine for multi-agent competitive analysis"
)

# Enable CORS for React frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "https://nexus-clash.vercel.app",
    "https://debate-ui-bgo9.vercel.app",
    "http://localhost:5173",
     ],
    allow_credentials=False,   
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# Retrieve Gemini API Key from environment variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Configure Gemini Client
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class DebateRequest(BaseModel):
    topic: str
    agent_a_profile: str = "Techno-Optimist"
    agent_b_profile: str = "Cautious Tech-Skeptic"
    rounds: int = 3

async def run_debate_lifecycle(topic: str, agent_a: str, agent_b: str, total_rounds: int) -> AsyncGenerator[str, None]:
    """
    State machine that runs a structured, turn-based debate.
    Each agent acts sequentially, receiving the full conversation history.
    Outputs Server-Sent Events (SSE) containing state-transitions and streamed content.
    """
    if not GEMINI_API_KEY:
        yield "data: " + json.dumps({
            "event": "error", 
            "message": "GEMINI_API_KEY is missing on the server. Please export it before launching."
        }) + "\n\n"
        return

    # Using the fast & highly capable reasoning preview model
    model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
    
    # Store transcript history to pass as context
    debate_history = []

    # Send INITIALIZATION event
    yield "data: " + json.dumps({
        "event": "start", 
        "topic": topic,
        "agent_a": agent_a,
        "agent_b": agent_b,
        "total_rounds": total_rounds
    }) + "\n\n"
    await asyncio.sleep(0.5)

    # Begin Turn-Based Debate Loop
    for r in range(1, total_rounds + 1):
        
        # ----------------------------------------------------
        # TURN 1: AGENT A'S TURN
        # ----------------------------------------------------
        yield "data: " + json.dumps({
            "event": "status", 
            "message": f"Agent A is analyzing the board and formulating Round {r} argument..."
        }) + "\n\n"
        await asyncio.sleep(0.2)

        # Build context for Agent A (PRO)
        history_str = "\n\n".join([
            f"Round {item['round']} | Agent {item['agent']} ({agent_a if item['agent'] == 'A' else agent_b}): {item['text']}" 
            for item in debate_history
        ])

        prompt_a = (
            f"You are Agent A, an elite debater arguing IN FAVOR of: '{topic}'.\n"
            f"Your specific persona is: {agent_a}.\n"
            f"Strictly adopt this character. Address and systematically dismantle any previous counterpoints.\n"
            f"Keep your response razor-sharp, highly logical, professional, and under 150 words. This is Round {r} of {total_rounds}.\n\n"
            f"--- DEBATE HISTORY SO FAR ---\n"
            f"{history_str if history_str else '[This is your opening constructive statement. Establish your thesis clearly.]'}\n\n"
            f"Now, deliver your argument:"
        )

        # Signal streaming start for Agent A
        yield "data: " + json.dumps({"event": "stream_start", "agent": "A", "round": r}) + "\n\n"

        try:
            # Call Gemini in streaming mode
            response = await asyncio.to_thread(
                model.generate_content,
                prompt_a,
                stream=True
            )
            
            accumulated_a_text = ""
            for chunk in response:
                if chunk.text:
                    accumulated_a_text += chunk.text
                    yield "data: " + json.dumps({
                        "event": "stream_chunk", 
                        "agent": "A", 
                        "text": chunk.text
                    }) + "\n\n"
                    # Small sleep to maintain uniform, legible streaming output
                    await asyncio.sleep(0.01)

            # Record Agent A's response in history
            debate_history.append({"round": r, "agent": "A", "text": accumulated_a_text})
            
            # Close stream block for Agent A
            yield "data: " + json.dumps({
                "event": "stream_end", 
                "agent": "A", 
                "round": r, 
                "full_text": accumulated_a_text
            }) + "\n\n"

        except Exception as e:
            yield "data: " + json.dumps({"event": "error", "message": f"Agent A generation failed: {str(e)}"}) + "\n\n"
            return

        await asyncio.sleep(1.0)

        # ----------------------------------------------------
        # TURN 2: AGENT B'S TURN
        # ----------------------------------------------------
        yield "data: " + json.dumps({
            "event": "status", 
            "message": f"Agent B is scanning Agent A's arguments and framing Round {r} counter-rebuttal..."
        }) + "\n\n"
        await asyncio.sleep(0.2)

        # Re-build updated context for Agent B (CON)
        history_str = "\n\n".join([
            f"Round {item['round']} | Agent {item['agent']} ({agent_a if item['agent'] == 'A' else agent_b}): {item['text']}" 
            for item in debate_history
        ])

        prompt_b = (
            f"You are Agent B, an elite debater arguing critically AGAINST the topic: '{topic}'.\n"
            f"Your specific persona is: {agent_b}.\n"
            f"Directly analyze, attack, and exploit logical vulnerabilities in Agent A's arguments.\n"
            f"Keep your response brilliant, persuasive, critical, and under 150 words. This is Round {r} of {total_rounds}.\n\n"
            f"--- DEBATE HISTORY SO FAR ---\n"
            f"{history_str}\n\n"
            f"Now, deliver your counterpoint:"
        )

        # Signal streaming start for Agent B
        yield "data: " + json.dumps({"event": "stream_start", "agent": "B", "round": r}) + "\n\n"

        try:
            response = await asyncio.to_thread(
                model.generate_content,
                prompt_b,
                stream=True
            )
            
            accumulated_b_text = ""
            for chunk in response:
                if chunk.text:
                    accumulated_b_text += chunk.text
                    yield "data: " + json.dumps({
                        "event": "stream_chunk", 
                        "agent": "B", 
                        "text": chunk.text
                    }) + "\n\n"
                    await asyncio.sleep(0.01)

            debate_history.append({"round": r, "agent": "B", "text": accumulated_b_text})
            
            # Close stream block for Agent B
            yield "data: " + json.dumps({
                "event": "stream_end", 
                "agent": "B", 
                "round": r, 
                "full_text": accumulated_b_text
            }) + "\n\n"

        except Exception as e:
            yield "data: " + json.dumps({"event": "error", "message": f"Agent B generation failed: {str(e)}"}) + "\n\n"
            return

        await asyncio.sleep(1.0)

    # ----------------------------------------------------
    # PHASE 3: THE JUDGE
    # ----------------------------------------------------
    yield "data: " + json.dumps({
        "event": "status", 
        "message": "The Debate has concluded. Transferring transcripts to the Supreme AI Judge..."
    }) + "\n\n"
    await asyncio.sleep(1.5)

    transcript_summary = "\n\n".join([
        f"Round {item['round']} | Agent {item['agent']} ({agent_a if item['agent'] == 'A' else agent_b}):\n{item['text']}" 
        for item in debate_history
    ])

    judge_prompt = (
        f"You are the Supreme Court AI Judge, an objective, hyper-rational adjudicator.\n"
        f"Analyze the following {total_rounds}-round transcript of a debate on: '{topic}'.\n\n"
        f"--- FULL DEBATE TRANSCRIPT ---\n"
        f"{transcript_summary}\n\n"
        f"Provide a definitive and deeply reasoned verdict formatted in elegant, structured Markdown.\n"
        f"You MUST format the output to contain these precise sections:\n"
        f"1. # Executive Verdict (Explicitly state the winner, a quantitative score like 88-84, and the core reasoning)\n"
        f"2. ## Agent A Arguments appraisal (Highlighting strengths and missed opportunities)\n"
        f"3. ## Agent B Arguments appraisal (Highlighting strengths and missed opportunities)\n"
        f"4. ## Rebuttal Clashes Analysis (Analyse how well they actually listened and counter-attacked)\n"
        f"5. ## Scoring Breakdown (A Markdown Table covering: Logical Rigor, Rhetoric, Persuasiveness, Rebuttals)\n"
        f"6. ## Core Post-Debate Synthesis (What key underlying truth does this clash reveal?)"
    )

    yield "data: " + json.dumps({"event": "judge_start"}) + "\n\n"

    try:
        response = await asyncio.to_thread(
            model.generate_content,
            judge_prompt,
            stream=True
        )

        for chunk in response:
            if chunk.text:
                yield "data: " + json.dumps({
                    "event": "judge_chunk", 
                    "text": chunk.text
                }) + "\n\n"
                await asyncio.sleep(0.005)

    except Exception as e:
        yield "data: " + json.dumps({"event": "error", "message": f"Judge assessment failed: {str(e)}"}) + "\n\n"
        return

    # Terminate SSE transmission
    yield "data: " + json.dumps({"event": "complete"}) + "\n\n"


@app.post("/api/debate/stream")
async def debate_stream_endpoint(request: DebateRequest):
    """
    HTTP POST Endpoint returning standard SSE (Server-Sent Events) streams.
    Allows passing complex parameters cleanly.
    """
    return StreamingResponse(
        run_debate_lifecycle(
            topic=request.topic,
            agent_a=request.agent_a_profile,
            agent_b=request.agent_b_profile,
            total_rounds=request.rounds
        ),
                media_type="text/event-stream"
    )


@app.get("/")
async def root():
    return {"status": "Aegis Debate Engine is running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "gemini_key_set": bool(GEMINI_API_KEY)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
