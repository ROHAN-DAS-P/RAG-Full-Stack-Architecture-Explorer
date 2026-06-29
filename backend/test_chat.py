import asyncio
import json
import websockets

async def test_rag():
    uri = "ws://localhost:8000/ws/chat"
    
    # Connect to the FastAPI WebSocket
    async with websockets.connect(uri) as websocket:
        question = input("Ask a question about your documents: ")
        
        # 1. Send the query matching your ws_protocol.py schema
        request = {"type": "query", "payload": {"text": question}}
        await websocket.send(json.dumps(request))
        
        print("\nAI Thinking...\n")
        
        # 2. Listen for the streaming response
        while True:
            response_raw = await websocket.recv()
            response = json.loads(response_raw)
            
            msg_type = response.get("type")
            
            if msg_type == "citations":
                citations = response["payload"]["citations"]
                if citations:
                    print(f"[Found {len(citations)} sources in your database]")
            
            elif msg_type == "token":
                # Print tokens as they arrive without adding new lines
                print(response["payload"]["text"], end="", flush=True)
                
            elif msg_type == "done":
                print("\n\n[Stream finished]")
                break
                
            elif msg_type == "error":
                print(f"\n[ERROR]: {response['payload']['message']}")
                break

if __name__ == "__main__":
    asyncio.run(test_rag())