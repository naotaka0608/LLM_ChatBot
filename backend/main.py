from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from typing import Optional, Dict, List
import logging
from pathlib import Path
import shutil

from rag_manager import RAGManager

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

# RAGマネージャー初期化
rag_manager = RAGManager()

class ChatRequest(BaseModel):
    message: str
    model_id: Optional[str] = "gpt2"
    max_length: Optional[int] = 100
    temperature: Optional[float] = 0.7
    use_rag: Optional[bool] = False
    rag_k: Optional[int] = 3

class ModelSelectRequest(BaseModel):
    model_id: str

@app.get("/")
async def root():
    return {
        "message": "HuggingFace LLM Chatbot API with RAG",
        "endpoints": {
            "/models": "利用可能なモデル一覧",
            "/chat": "チャット（POST）",
            "/model/select": "モデル選択（POST）",
            "/model/current": "現在のモデル情報",
            "/documents/upload": "ドキュメントアップロード（POST）",
            "/documents/list": "ドキュメント一覧（GET）",
            "/documents/delete": "全ドキュメント削除（DELETE）"
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
    """チャットメッセージを処理（RAG対応）"""
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

        # RAG使用時は関連ドキュメントを検索
        retrieved_docs = []
        if request.use_rag:
            logger.info(f"RAG検索を実行: k={request.rag_k}")
            retrieved_docs = rag_manager.search(request.message, k=request.rag_k)

            if retrieved_docs:
                # RAG有効時は検索結果を直接まとめて返す
                logger.info(f"{len(retrieved_docs)}件のドキュメントを発見")

                # 検索結果をまとめて回答を生成
                answer_parts = []
                answer_parts.append(f"アップロードされたドキュメントから{len(retrieved_docs)}件の関連情報が見つかりました：\n")

                for i, doc in enumerate(retrieved_docs, 1):
                    source = doc['source']
                    content = doc['content'][:300]  # 最初の300文字
                    answer_parts.append(f"\n【{i}. {source}】")
                    answer_parts.append(content)
                    if len(doc['content']) > 300:
                        answer_parts.append("...")

                # 検索ベースの回答を直接返す（LLM生成なし）
                response_text = "\n".join(answer_parts)

                response_data = {
                    "status": "success",
                    "response": response_text,
                    "model_id": model_id,
                    "input": request.message,
                    "retrieved_docs": retrieved_docs,
                    "context_used": True,
                    "rag_direct_answer": True  # RAGの直接回答であることを示す
                }

                return response_data
            else:
                # 検索結果がない場合
                response_data = {
                    "status": "success",
                    "response": "アップロードされたドキュメントから関連する情報が見つかりませんでした。",
                    "model_id": model_id,
                    "input": request.message,
                    "retrieved_docs": [],
                    "context_used": False
                }

                return response_data

        # RAG未使用時は通常のLLM生成
        prompt = request.message

        # テキスト生成
        logger.info(f"テキスト生成中: {prompt[:100]}...")

        result = pipe(
            prompt,
            max_new_tokens=150,
            temperature=request.temperature,
            num_return_sequences=1,
            do_sample=True,
            pad_token_id=pipe.tokenizer.eos_token_id,
            eos_token_id=pipe.tokenizer.eos_token_id,
            repetition_penalty=1.2
        )

        generated_text = result[0]["generated_text"]

        response_data = {
            "status": "success",
            "response": generated_text,
            "model_id": model_id,
            "input": request.message
        }

        return response_data

    except Exception as e:
        logger.error(f"チャット処理エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=f"チャット処理に失敗しました: {str(e)}")

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """ドキュメントをアップロード"""
    try:
        # ファイル拡張子チェック
        filename = file.filename
        if not filename:
            raise HTTPException(status_code=400, detail="ファイル名が不正です")

        file_ext = Path(filename).suffix.lower()
        if file_ext not in ['.pdf', '.txt', '.md']:
            raise HTTPException(
                status_code=400,
                detail=f"未対応のファイル形式です: {file_ext}。対応形式: .pdf, .txt, .md"
            )

        # ファイルを保存
        file_path = rag_manager.upload_dir / filename
        with open(file_path, 'wb') as f:
            shutil.copyfileobj(file.file, f)

        # ドキュメントを処理（埋め込み生成）
        logger.info(f"ドキュメント処理開始: {filename}")
        doc_info = rag_manager.process_document(file_path, filename)

        return {
            "status": "success",
            "message": f"ファイル '{filename}' をアップロードしました",
            "document": doc_info
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ファイルアップロードエラー: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ファイルアップロードに失敗しました: {str(e)}")

@app.get("/documents/list")
async def list_documents():
    """アップロード済みドキュメントの一覧を取得"""
    try:
        documents = rag_manager.get_documents()
        return {
            "status": "success",
            "count": len(documents),
            "documents": documents
        }
    except Exception as e:
        logger.error(f"ドキュメント一覧取得エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ドキュメント一覧の取得に失敗しました: {str(e)}")

@app.delete("/documents/delete")
async def delete_all_documents():
    """全てのドキュメントを削除"""
    try:
        rag_manager.delete_all_documents()
        return {
            "status": "success",
            "message": "全てのドキュメントを削除しました"
        }
    except Exception as e:
        logger.error(f"ドキュメント削除エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ドキュメント削除に失敗しました: {str(e)}")

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    documents = rag_manager.get_documents()
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "loaded_models": list(model_cache.keys()),
        "current_model": current_model_id,
        "rag_enabled": True,
        "documents_count": len(documents)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
