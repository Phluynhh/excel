import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from agents.orchestrator import Orchestrator
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app)
orchestrator = Orchestrator()

@app.route('/api/upload_and_embed', methods=['POST'])
def upload_and_embed():
    file = request.files.get("file")
    config = request.form.get("config")

    if not file or not config:
        return jsonify({"error": "File or config missing"}), 400

    import json
    config = json.loads(config)

    # Save file to /files
    save_path = os.path.join("files", file.filename)
    file.save(save_path)

    # Gọi xử lý embedding với file vừa lưu
    from utils.prepare_embeddings import prepare_and_store_embeddings
    prepare_and_store_embeddings(save_path, config)

    return jsonify({"message": "Embedding done!"})


@app.route('/api/query', methods=['POST'])
def query():
    data = request.get_json()
    query = data.get('query')
    context = data.get('context', "")  # Lấy context từ request, nếu không có sẽ mặc định là chuỗi rỗng

    if not query:
        return jsonify({"error": "No query provided!"}), 400

    # Gọi handle_query với đủ hai đối số query và context
    response = orchestrator.handle_query(query, context)
    return jsonify({"response": response})

@app.route("/", methods=["GET"])
def health_check():
    return "API is running"

if __name__ == '__main__':
    app.run(debug=True)
