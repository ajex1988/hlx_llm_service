from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_csp.csp import csp_header

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import base64
from io import BytesIO
import argparse

# --- Parse CLI args ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen-VL-Chat')
    parser.add_argument('--port', type=int, default=3097)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_new_tokens', type=int, default=2048)
    return parser.parse_args()

# --- Decode base64 image ---
def decode_base64_image(base64_str):
    try:
        header, encoded = base64_str.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        return Image.open(BytesIO(img_bytes))
    except Exception as e:
        print("Failed to decode image:", e)
        return None

# --- Main ---
if __name__ == '__main__':
    args = parse_args()

    # Initialize model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map=args.device,
        trust_remote_code=True
    ).eval()

    # Init Flask
    app = Flask(__name__)
    CORS(app)

    @app.route('/vlm_chat', methods=['POST'])
    @csp_header({'default-src': "'self'", 'script-src': "'self'"})
    def chat():
        try:
            data = request.get_json()
            input_text = data.get('text', '')
            image_base64 = data.get('image', None)

            # Build prompt
            query_items = []
            if image_base64:
                image = decode_base64_image(image_base64)
                if image:
                    query_items.append({'image': image})
            query_items.append({'text': input_text})
            prompt = tokenizer.from_list_format(query_items)

            # Run inference
            response, _ = model.chat(tokenizer, query=prompt, history=None)

            return jsonify({'reply': response})
        except Exception as e:
            print("Server error:", e)
            return jsonify({'reply': '[模型服务出错]'}), 500

    print(f"🚀 VLM Server ready on port {args.port}")
    app.run(host='0.0.0.0', port=args.port)
