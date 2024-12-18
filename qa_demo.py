from flask import Flask, jsonify, request
import threading
import torch

from flask_cors import CORS
from flask_csp.csp import csp_header
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./', help='Model Directory')
    parser.add_argument('--port', type=int, default=3096)
    parser.add_argument('--max_concurrent', type=int, default=5)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    model_path = args.model_path
    MAX_CONCURRENT_REQUESTS = args.max_concurrent
    port = args.port

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

    lock = threading.Lock()
    counter = 0

    app = Flask(__name__)
    CORS(app)

    @app.route('/hlx_llm_service', methods=['POST'])
    @csp_header({'default-src': "'self'", 'script-src': "'self'"})
    def hlx_llm_service():
        global counter

        if counter >= MAX_CONCURRENT_REQUESTS:
            return jsonify({'error': 'Too many requests'})

        with lock:
            counter += 1

        try:
            question = request.json['question']
            question += "->"
            inputs = tokenizer(question, return_tensors='pt')
            inputs = inputs.to(device)
            pred = model.generate(**inputs)

            text = tokenizer.decode(pred.cpu().detach().numpy()[0])

            print(f"prediction: {text}")

            response = {'answer': text[len(question):]}
            return jsonify(response)
        finally:
            with lock:
                counter -= 1

    print("Flask Server Started")
    app.run(host='127.0.0.1', port=port)


