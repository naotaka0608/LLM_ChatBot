from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from typing import Optional, Dict
import logging

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS設定（Electronからのアクセスを許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 利用可能なモデルリスト（軽量で日本語対応のモデルを中心に）
AVAILABLE_MODELS = {
    "rinna-japanese-gpt2": {
        "name": "rinna/japanese-gpt2-medium",
        "description": "日本語GPT-2モデル（軽量）"
    },
    "gpt2": {
        "name": "gpt2",
        "description": "GPT-2（英語、軽量）"
    },
    "gpt2-medium": {
        "name": "gpt2-medium",
        "description": "GPT-2 Medium（英語）"
    },
    "rinna-japanese-gpt-1b": {
        "name": "rinna/japanese-gpt-1b",
        "description": "日本語GPT 1Bモデル"
    },
    "cyberagent-calm2": {
        "name": "cyberagent/calm2-7b-chat",
        "description": "CyberAgent CALM2 7B Chat（日本語、大きめ）"
    }
}

# モデルキャッシュ
model_cache: Dict[str, Dict] = {}
current_model_id: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    model_id: Optional[str] = "gpt2"
    max_length: Optional[int] = 100
    temperature: Optional[float] = 0.7

class ModelSelectRequest(BaseModel):
    model_id: str

@app.get("/")
async def root():
    return {
        "message": "HuggingFace LLM Chatbot API",
        "endpoints": {
            "/models": "利用可能なモデル一覧",
            "/chat": "チャット（POST）",
            "/model/select": "モデル選択（POST）",
            "/model/current": "現在のモデル情報"
        }
    }

@app.get("/models")
async def get_models():
    """利用可能なモデルの一覧を返す"""
    return {
        "models": AVAILABLE_MODELS,
        "current_model": current_model_id
    }

@app.get("/model/current")
async def get_current_model():
    """現在ロードされているモデル情報を返す"""
    if current_model_id:
        return {
            "model_id": current_model_id,
            "model_info": AVAILABLE_MODELS.get(current_model_id, {})
        }
    return {"model_id": None, "message": "モデルがロードされていません"}

@app.post("/model/select")
async def select_model(request: ModelSelectRequest):
    """モデルを選択してロード"""
    global current_model_id

    if request.model_id not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail="指定されたモデルIDが見つかりません")

    try:
        logger.info(f"モデルをロード中: {request.model_id}")
        load_model(request.model_id)
        current_model_id = request.model_id

        return {
            "status": "success",
            "model_id": current_model_id,
            "model_info": AVAILABLE_MODELS[current_model_id]
        }
    except Exception as e:
        logger.error(f"モデルのロードに失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"モデルのロードに失敗しました: {str(e)}")

def load_model(model_id: str):
    """モデルをロードしてキャッシュに保存"""
    global model_cache

    if model_id in model_cache:
        logger.info(f"キャッシュからモデルを使用: {model_id}")
        return

    model_name = AVAILABLE_MODELS[model_id]["name"]
    logger.info(f"モデルをダウンロード中: {model_name}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )

        # pipelineの作成
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )

        model_cache[model_id] = {
            "tokenizer": tokenizer,
            "model": model,
            "pipeline": pipe
        }

        logger.info(f"モデルのロード完了: {model_id}")

    except Exception as e:
        logger.error(f"モデルのロードエラー: {str(e)}")
        raise

@app.post("/chat")
async def chat(request: ChatRequest):
    """チャットメッセージを処理"""
    global current_model_id

    # モデルが指定されていない場合、デフォルトを使用
    model_id = request.model_id

    if model_id not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail="指定されたモデルIDが見つかりません")

    try:
        # モデルをロード（キャッシュされていなければ）
        if model_id not in model_cache:
            logger.info(f"モデルを初回ロード: {model_id}")
            load_model(model_id)
            current_model_id = model_id

        pipe = model_cache[model_id]["pipeline"]

        # テキスト生成
        logger.info(f"テキスト生成中: {request.message[:50]}...")

        result = pipe(
            request.message,
            max_length=request.max_length,
            temperature=request.temperature,
            num_return_sequences=1,
            do_sample=True,
            pad_token_id=pipe.tokenizer.eos_token_id
        )

        generated_text = result[0]["generated_text"]

        return {
            "status": "success",
            "response": generated_text,
            "model_id": model_id,
            "input": request.message
        }

    except Exception as e:
        logger.error(f"チャット処理エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=f"チャット処理に失敗しました: {str(e)}")

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "loaded_models": list(model_cache.keys()),
        "current_model": current_model_id
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
