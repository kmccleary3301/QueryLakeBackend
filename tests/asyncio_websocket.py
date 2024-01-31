import asyncio
import websockets
import json
import time

input = {
    "auth": {
        "username": "4ccbbe2d-2734-4967-ada0-b23eea46",
        "password_prehash": "c76173ef1b26130cdb9406ce92dfb45a521130b00272a0dd3298c2d8e688b31e"
    }
}
input.update({
    "command" : "toolchain/create",
    "arguments": {
        "toolchain_id": "chat_session_normal"
    }
})
import asyncio
import websockets
import json
import time

async def handle_message(message):
    # This function will be called every time a message is received
    print(f"Received message: {message}")

async def websocket_client(uri, handle_message):
    async with websockets.connect(uri) as websocket:
        # Save the websocket in a global variable
        global websocket_global
        websocket_global = websocket

        # Process incoming messages
        async for message in websocket:
            await handle_message(message)

async def send_message(websocket, message):
    await websocket.send(message)

# Start the WebSocket client as a separate task
asyncio.run(websocket_client('ws://localhost:8000/toolchain', handle_message))

# Wait for the WebSocket client to connect
time.sleep(1)

# Send a message
asyncio.run(send_message(websocket_global, json.dumps(input)))

# Your script can continue running here