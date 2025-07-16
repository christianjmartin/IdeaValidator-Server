from flask import Flask, request, Response
from flask_cors import CORS
from agent import run_agent_streaming

app = Flask(__name__)
CORS(app)

@app.route('/api/ask', methods=['POST'])
def ask():
    data = request.get_json()
    new_message = data.get('message')

    conversation = data.get("conversation")
    all_messages = conversation.get("messages", [])

    # format context in user - assistant pairs
    context = [
        {
            "role": "user" if msg["sender"] == "user" else "assistant",
            "content": msg["text"].strip()
        }
        for msg in all_messages
        if msg.get("text", "").strip() != ""
    ]

    def generate():
        try:
            # raise Exception("test")
            for token in run_agent_streaming(new_message, context):
                yield token
        except Exception as e:
            yield '<span style="color: red;"> There was an error when gathering my response.</span>'
            

    return Response(generate(), mimetype='text/plain')

if __name__ == '__main__':
    print("flask server started")
    app.run(port=8080, threaded=True)
