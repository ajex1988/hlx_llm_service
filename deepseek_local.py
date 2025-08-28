import logging
import json
import requests
from flask import Flask, render_template, request, Response, jsonify
from flask_cors import CORS

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ollama_generate.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Flask应用初始化
app = Flask(__name__)
CORS(app)
app.config['JSON_AS_ASCII'] = False  # 支持中文显示

# Ollama配置
OLLAMA_URL = "http://192.168.10.16:11434/api/generate"  # Generate API地址
MODEL_NAME = "deepseek-r1:70b"  # 使用的模型
DEFAULT_OPTIONS = {
    "temperature": 0.7,
    "max_tokens": 2048,
    "stream": True
}


@app.route('/')
def index():
    """渲染首页"""
    logger.info("用户访问首页")
    return render_template('index.html')


def stream_ollama_generate(prompt: str, options: dict = None):
    """调用Ollama的Generate API，流式返回结果"""
    # 构建请求数据
    request_data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "options": options or DEFAULT_OPTIONS,
        "stream": True
    }

    logger.info(f"调用Generate API | prompt: {prompt[:50]}...")

    try:
        # 发送请求到Ollama
        with requests.post(
                OLLAMA_URL,
                json=request_data,
                stream=True,
                timeout=360
        ) as response:
            if not response.ok:
                error_msg = f"Ollama服务错误 | 状态码: {response.status_code} | 内容: {response.text}"
                logger.error(error_msg)
                yield f"data: {json.dumps({'type': 'error', 'value': error_msg})}\n\n"
                return

            # 处理流式响应
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue

                try:
                    generate_data = json.loads(line)

                    # 处理生成的内容片段
                    if "response" in generate_data and generate_data["response"]:
                        chunk = generate_data["response"]
                        yield f"data: {json.dumps({'type': 'chunk', 'value': chunk})}\n\n"

                    # 处理生成结束标记
                    if generate_data.get("done", False):
                        logger.info(f"生成完成 | 总tokens: {generate_data.get('total_tokens', 0)}")
                        yield f"data: {json.dumps({'type': 'done', 'value': '生成完成'})}\n\n"
                        break

                except json.JSONDecodeError as e:
                    error_msg = f"解析响应失败 | 原始数据: {line} | 错误: {str(e)}"
                    logger.error(error_msg)
                    yield f"data: {json.dumps({'type': 'error', 'value': error_msg})}\n\n"
                    break

    except requests.exceptions.RequestException as e:
        error_msg = f"连接Ollama失败 | 错误: {str(e)} | 检查地址: {OLLAMA_URL}"
        logger.error(error_msg)
        yield f"data: {json.dumps({'type': 'error', 'value': error_msg})}\n\n"
    except Exception as e:
        error_msg = f"生成过程出错 | {str(e)}"
        logger.error(error_msg, exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'value': error_msg})}\n\n"


@app.route('/generate', methods=['POST'])
def generate():
    """处理生成请求的接口"""
    try:
        # 获取请求数据
        request_data = request.get_json()
        if not request_data or "prompt" not in request_data:
            logger.warning("请求缺少prompt参数")
            return jsonify({"error": "请提供生成提示词"}), 400

        prompt = request_data["prompt"].strip()
        if not prompt:
            logger.warning("用户提交了空提示词")
            return jsonify({"error": "提示词不能为空"}), 400

        # 处理自定义参数
        custom_options = request_data.get("options", {})
        final_options = {**DEFAULT_OPTIONS, **custom_options}

        # 返回流式响应
        return Response(
            stream_ollama_generate(prompt, final_options),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive'
            }
        )

    except Exception as e:
        error_msg = f"处理生成请求失败 | {str(e)}"
        logger.error(error_msg, exc_info=True)
        return jsonify({"error": error_msg}), 500


if __name__ == '__main__':
    logger.info(f"启动Ollama Generate服务 | 模型: {MODEL_NAME} | 地址: {OLLAMA_URL}")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
