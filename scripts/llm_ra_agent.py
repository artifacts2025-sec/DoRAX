import base64
import cv2
import pyttsx3
import os
from dotenv import load_dotenv, find_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

import whisper
import webrtcvad
import sounddevice as sd
import collections
import queue
import numpy as np
import sys
import torch

from policy_loader import load_dorax_policies
'''
Note this script is only for showing the build-up process of the llm ra agent using langchain.
In practical execution, the policy loader should be replaced with real ACT model and config with the RA device.
'''
load_dotenv(find_dotenv())
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set your OPENAI_API_KEY in .env file or environment variables.")

engine = pyttsx3.init()

# ----- Policy loading -----
print("loading llm ra agent policies...")
policy_base_path = "../example/policy"
policy_loader = load_dorax_policies(policy_base_path)

print(f"available policies: {policy_loader.list_policies()}")

# ----- ONLY FOR SIMULATION -----
@tool(return_direct=True)
def pick_sponge():
    """genetly pick up the sponge and put it on the container"""
    print("[POLICY] Executing pick_sponge action using softhandling policy...")
    try:
        mock_observation = {
            "observation.images.image": torch.randn(1, 3, 224, 224),
            "observation.images.wrist_image": torch.randn(1, 3, 224, 224),
            "observation.state": torch.randn(1, 7)
        }
        
        action = policy_loader.execute_policy("softhandling", mock_observation)
        print(f"Policy action shape: {action.shape}")
        return f"Done executing pick_sponge action using softhandling policy. Action: {action.flatten()[:3].tolist()}"
    except Exception as e:
        print(f"Error executing policy: {e}")
        return "Done grabbing sponge (fallback)."

@tool(return_direct=True)
def sort_screws():
    """Pick up the black screws and put them in the right container,pick up the gray screws and put them in the left container"""
    print("[POLICY] Executing sort_screws action using sorting policy...")
    try:
        mock_observation = {
            "observation.images.image": torch.randn(1, 3, 224, 224),
            "observation.images.wrist_image": torch.randn(1, 3, 224, 224),
            "observation.state": torch.randn(1, 7)
        }
        
        action = policy_loader.execute_policy("sorting", mock_observation)
        print(f"Policy action shape: {action.shape}")
        return f"Done executing sort_screws action using sorting policy. Action: {action.flatten()[:3].tolist()}"
    except Exception as e:
        print(f"Error executing policy: {e}")
        return "Done grabbing orange (fallback)."

@tool(return_direct=True)
def transfer_box():
    """pick up the box and move it slowly to the target location"""
    print("[POLICY] Executing transfer_box action using transfer policy...")
    try:
        mock_observation = {
            "observation.images.image": torch.randn(1, 3, 224, 224),
            "observation.images.wrist_image": torch.randn(1, 3, 224, 224),
            "observation.state": torch.randn(1, 7)
        }
        
        action = policy_loader.execute_policy("transfer", mock_observation)
        print(f"Policy action shape: {action.shape}")
        return f"Done executing transfer_box action using transfer policy. Action: {action.flatten()[:3].tolist()}"
    except Exception as e:
        print(f"Error executing policy: {e}")
        return "Done grabbing candy (fallback)."



@tool(return_direct=True)
def describe_area():
    """Describing what I can see."""
    img_path = "sample_table.png"
    img = cv2.imread(img_path)
    if img is None:
        return "No image found to describe."
    _, encoded_img = cv2.imencode('.png', img)
    base64_img = base64.b64encode(encoded_img).decode("utf-8")
    mime_type = 'image/png'
    encoded_image_url = f"data:{mime_type};base64,{base64_img}"
    llm = ChatOpenAI(temperature=0.1, model="gpt-4o-mini", api_key=api_key)
    chat_prompt_template = ChatPromptTemplate.from_messages(
        messages=[
            SystemMessage(content='Describe in one phrase what objects you see on the table. Not including robot. Start answer with "I see..."'),
            HumanMessagePromptTemplate.from_template([{'image_url': "{encoded_image_url}", 'type': 'image_url'}])
        ]
    )
    chain = chat_prompt_template | llm
    res = chain.invoke({"encoded_image_url": encoded_image_url})
    return res.content

agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You're a helpful robot-arm assistant. Answer super concise."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

tools = [pick_sponge, sort_screws, transfer_box, describe_area]

llm = ChatOpenAI(temperature=0.1, model="gpt-4o-mini", api_key=api_key)
agent = create_tool_calling_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- Whisper + VAD  ---
model = whisper.load_model("base")
sample_rate = 16000
frame_duration_ms = 30
frame_size = int(sample_rate * frame_duration_ms / 1000)
vad = webrtcvad.Vad(2) 

q = queue.Queue()

def audio_callback(indata, frames, time_, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

def frame_generator():
    while True:
        frame = q.get()
        yield frame

def vad_collector():
    frames = frame_generator()
    ring_buffer = collections.deque(maxlen=10)
    triggered = False
    voiced_frames = []

    for frame in frames:
        is_speech = vad.is_speech(frame, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield b"".join(voiced_frames)
                ring_buffer.clear()
                voiced_frames = []

def listen():
    print("Listening for commands, please speak clearly...")
    with sd.RawInputStream(samplerate=sample_rate, blocksize=frame_size, dtype='int16',
                           channels=1, callback=audio_callback):
        for segment in vad_collector():
            print("Processing...")
            audio_data = np.frombuffer(segment, dtype=np.int16).astype(np.float32) / 32768.0
            result = model.transcribe(audio_data, fp16=False)
            text = result["text"].strip()
            if text:
                print(f"Recognized: {text}")
                return text
            else:
                print("No speech recognized, continue listening...")

def generate_response(prompt):
    completions = agent_executor.invoke({"input": prompt})
    return completions["output"]

def main():
    print("Starting simulated LLM Agent. Say commands like 'grab sponge', 'describe area', or 'thank you' to exit.")
    try:
        while True:
            audio_prompt = listen()
            if audio_prompt is None:
                print("Could not understand. Please repeat.")
                continue

            print(f"You: {audio_prompt}")
            response = generate_response(audio_prompt)
            print(f"Robot: {response}")
            engine.say(response)
            engine.runAndWait()

            if audio_prompt.strip().lower() == "thank you":
                print("Exiting...")
                break
    except KeyboardInterrupt:
        print("\nUser interrupted. Exiting...")

if __name__ == "__main__":
    main()
