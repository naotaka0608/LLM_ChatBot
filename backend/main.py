from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from typing import Optional, Dict, List
import logging
from pathlib import Path
import shutil
import uuid
import secrets

from rag_manager import RAGManager
from db_manager import DBManager

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

# データベースマネージャー初期化
db_manager = DBManager(db_path="./chatbot.db")

# RAGマネージャー初期化（DBマネージャーを渡す）
rag_manager = RAGManager(db_manager=db_manager)

# セッショントークンの保存（本番環境ではRedisなどを使用すべき）
active_sessions: Dict[str, str] = {}  # token -> username

# 認証関数
async def get_current_user(authorization: Optional[str] = Header(None)) -> str:
    """
    認証トークンからユーザー名を取得

    Args:
        authorization: Authorizationヘッダー

    Returns:
        ユーザー名

    Raises:
        HTTPException: 認証失敗時
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="認証が必要です")

    token = authorization.replace("Bearer ", "")
    username = active_sessions.get(token)

    if not username:
        raise HTTPException(status_code=401, detail="無効なトークンです")

    return username

class SignUpRequest(BaseModel):
    username: str
    email: str
    password: str

class SignInRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    message: str
    model_id: Optional[str] = "gpt2"
    max_length: Optional[int] = 100
    temperature: Optional[float] = 0.7
    use_rag: Optional[bool] = False
    rag_k: Optional[int] = 3
    session_id: Optional[str] = None  # セッションID（省略時は自動生成）
    score_threshold: Optional[float] = None  # RAGスコア閾値（デフォルト: None=フィルタリングなし）

class ModelSelectRequest(BaseModel):
    model_id: str

@app.get("/")
async def root():
    return {
        "message": "HuggingFace LLM Chatbot API with RAG & Database & Authentication",
        "endpoints": {
            "/signup": "サインアップ（POST）",
            "/signin": "サインイン（POST）",
            "/signout": "サインアウト（POST）",
            "/me": "ユーザー情報取得（GET、認証必要）",
            "/models": "利用可能なモデル一覧",
            "/chat": "チャット（POST、認証必要）",
            "/model/select": "モデル選択（POST）",
            "/model/current": "現在のモデル情報",
            "/documents/upload": "ドキュメントアップロード（POST、認証必要）",
            "/documents/list": "ドキュメント一覧（GET）",
            "/documents/delete": "全ドキュメント削除（DELETE、認証必要）",
            "/chat/history/{session_id}": "セッション別チャット履歴（GET）",
            "/chat/history": "全チャット履歴（GET）",
            "/rag/searches": "RAG検索履歴（GET）",
            "/rag/popular-queries": "人気の検索クエリ（GET）",
            "/statistics": "データベース統計情報（GET）",
            "/health": "ヘルスチェック"
        }
    }

@app.post("/signup")
async def signup(request: SignUpRequest):
    """ユーザー登録"""
    try:
        # 入力バリデーション
        if len(request.username) < 3:
            raise HTTPException(status_code=400, detail="ユーザー名は3文字以上である必要があります")
        if len(request.password) < 6:
            raise HTTPException(status_code=400, detail="パスワードは6文字以上である必要があります")
        if "@" not in request.email:
            raise HTTPException(status_code=400, detail="有効なメールアドレスを入力してください")

        # ユーザー名の重複チェック
        existing_user = db_manager.get_user_by_username(request.username)
        if existing_user:
            raise HTTPException(status_code=400, detail="このユーザー名は既に使用されています")

        # メールアドレスの重複チェック
        existing_email = db_manager.get_user_by_email(request.email)
        if existing_email:
            raise HTTPException(status_code=400, detail="このメールアドレスは既に使用されています")

        # ユーザー作成
        user_id = db_manager.create_user(request.username, request.email, request.password)

        if user_id:
            # トークン生成
            token = secrets.token_urlsafe(32)
            active_sessions[token] = request.username

            return {
                "status": "success",
                "message": "ユーザー登録が完了しました",
                "token": token,
                "username": request.username
            }
        else:
            raise HTTPException(status_code=500, detail="ユーザー登録に失敗しました")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"サインアップエラー: {str(e)}")
        raise HTTPException(status_code=500, detail=f"サインアップに失敗しました: {str(e)}")

@app.post("/signin")
async def signin(request: SignInRequest):
    """ユーザー認証"""
    try:
        user = db_manager.verify_user(request.username, request.password)

        if user:
            # トークン生成
            token = secrets.token_urlsafe(32)
            active_sessions[token] = request.username

            return {
                "status": "success",
                "message": "サインインしました",
                "token": token,
                "username": user['username'],
                "email": user['email']
            }
        else:
            raise HTTPException(status_code=401, detail="ユーザー名またはパスワードが正しくありません")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"サインインエラー: {str(e)}")
        raise HTTPException(status_code=500, detail=f"サインインに失敗しました: {str(e)}")

@app.post("/signout")
async def signout(current_user: str = Depends(get_current_user), authorization: Optional[str] = Header(None)):
    """サインアウト"""
    try:
        if authorization:
            token = authorization.replace("Bearer ", "")
            if token in active_sessions:
                del active_sessions[token]

        return {
            "status": "success",
            "message": "サインアウトしました"
        }
    except Exception as e:
        logger.error(f"サインアウトエラー: {str(e)}")
        raise HTTPException(status_code=500, detail=f"サインアウトに失敗しました: {str(e)}")

@app.get("/me")
async def get_current_user_info(current_user: str = Depends(get_current_user)):
    """現在のユーザー情報を取得"""
    try:
        user = db_manager.get_user_by_username(current_user)
        if user:
            return {
                "status": "success",
                "user": user
            }
        else:
            raise HTTPException(status_code=404, detail="ユーザーが見つかりません")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ユーザー情報取得エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ユーザー情報の取得に失敗しました: {str(e)}")

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
async def chat(request: ChatRequest, current_user: str = Depends(get_current_user)):
    """チャットメッセージを処理（RAG対応、認証必要）"""
    global current_model_id

    # セッションIDの生成または取得
    session_id = request.session_id or str(uuid.uuid4())

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
            # メタデータのみの要求かどうかを検出
            # パターン1: 「ファイル名を列挙」のようにファイル名系 + 動詞の組み合わせ
            # パターン2: 「ファイル名だけ」「内容は表示しない」のような明示的な指示
            metadata_list_keywords = ["ファイル名", "ファイル一覧", "ドキュメント一覧"]
            action_keywords = ["列挙", "教えて", "見せて", "表示", "リスト"]
            no_content_keywords = ["内容は表示しない", "内容を表示しない", "ファイル名だけ", "ファイル名のみ"]

            has_metadata_keyword = any(keyword in request.message for keyword in metadata_list_keywords)
            has_action_keyword = any(keyword in request.message for keyword in action_keywords)
            has_no_content_keyword = any(keyword in request.message for keyword in no_content_keywords)

            # ファイル名系 + アクション、またはファイル名系 + 内容制限のいずれかでメタデータのみと判定
            is_metadata_only = (
                (has_metadata_keyword and has_action_keyword) or
                (has_metadata_keyword and has_no_content_keyword) or
                has_no_content_keyword
            )

            # メタデータのみの場合はドキュメント一覧を返す
            if is_metadata_only:
                logger.info("メタデータのみの要求を検出")
                all_docs = rag_manager.get_documents()

                if all_docs:
                    answer_parts = [f"アップロード済みのドキュメント（全{len(all_docs)}件）：\n"]
                    for i, doc in enumerate(all_docs, 1):
                        answer_parts.append(f"{i}. {doc['filename']}")
                    response_text = "\n".join(answer_parts)
                else:
                    response_text = "アップロードされたドキュメントはありません。"

                # チャット履歴をDBに保存
                db_manager.save_chat(
                    session_id=session_id,
                    user_message=request.message,
                    bot_response=response_text,
                    model_id=model_id,
                    use_rag=False,  # メタデータ取得のみなのでRAG検索ではない
                    retrieved_docs_count=len(all_docs) if all_docs else 0
                )

                return {
                    "status": "success",
                    "response": response_text,
                    "model_id": model_id,
                    "input": request.message,
                    "retrieved_docs": [],
                    "context_used": False,
                    "metadata_only": True,
                    "session_id": session_id
                }

            # 通常のRAG検索
            logger.info(f"RAG検索を実行: k={request.rag_k}, score_threshold={request.score_threshold}")
            retrieved_docs = rag_manager.search(
                request.message,
                k=request.rag_k,
                score_threshold=request.score_threshold
            )

            if retrieved_docs:
                # RAG有効時は検索結果を直接まとめて返す
                logger.info(f"{len(retrieved_docs)}件のドキュメントを発見")

                # 同じファイルからの重複を除去（最もスコアの高いもののみ残す）
                unique_docs = {}
                for doc in retrieved_docs:
                    source = doc['source']
                    if source not in unique_docs or doc['score'] < unique_docs[source]['score']:
                        unique_docs[source] = doc

                unique_docs_list = list(unique_docs.values())
                logger.info(f"重複除去後: {len(unique_docs_list)}件のユニークなドキュメント")

                # 検索結果をまとめて回答を生成
                answer_parts = []
                answer_parts.append(f"アップロードされたドキュメントから{len(unique_docs_list)}件の関連情報が見つかりました：\n")

                for i, doc in enumerate(unique_docs_list, 1):
                    source = doc['source']
                    content = doc['content'][:300]  # 最初の300文字
                    answer_parts.append(f"\n【{i}. {source}】")
                    answer_parts.append(content)
                    if len(doc['content']) > 300:
                        answer_parts.append("...")

                # 検索ベースの回答を直接返す（LLM生成なし）
                response_text = "\n".join(answer_parts)

                # チャット履歴をDBに保存
                db_manager.save_chat(
                    session_id=session_id,
                    user_message=request.message,
                    bot_response=response_text,
                    model_id=model_id,
                    use_rag=True,
                    retrieved_docs_count=len(unique_docs_list)
                )

                response_data = {
                    "status": "success",
                    "response": response_text,
                    "model_id": model_id,
                    "input": request.message,
                    "retrieved_docs": unique_docs_list,
                    "context_used": True,
                    "rag_direct_answer": True,  # RAGの直接回答であることを示す
                    "session_id": session_id
                }

                return response_data
            else:
                # 検索結果がない場合
                no_result_message = "アップロードされたドキュメントから関連する情報が見つかりませんでした。"

                # チャット履歴をDBに保存
                db_manager.save_chat(
                    session_id=session_id,
                    user_message=request.message,
                    bot_response=no_result_message,
                    model_id=model_id,
                    use_rag=True,
                    retrieved_docs_count=0
                )

                response_data = {
                    "status": "success",
                    "response": no_result_message,
                    "model_id": model_id,
                    "input": request.message,
                    "retrieved_docs": [],
                    "context_used": False,
                    "session_id": session_id
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

        # チャット履歴をDBに保存
        db_manager.save_chat(
            session_id=session_id,
            user_message=request.message,
            bot_response=generated_text,
            model_id=model_id,
            use_rag=False,
            retrieved_docs_count=0
        )

        response_data = {
            "status": "success",
            "response": generated_text,
            "model_id": model_id,
            "input": request.message,
            "session_id": session_id
        }

        return response_data

    except Exception as e:
        logger.error(f"チャット処理エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=f"チャット処理に失敗しました: {str(e)}")

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...), current_user: str = Depends(get_current_user)):
    """ドキュメントをアップロード（認証必要）"""
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
async def delete_all_documents(current_user: str = Depends(get_current_user)):
    """全てのドキュメントを削除（認証必要）"""
    try:
        rag_manager.delete_all_documents()
        return {
            "status": "success",
            "message": "全てのドキュメントを削除しました"
        }
    except Exception as e:
        logger.error(f"ドキュメント削除エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ドキュメント削除に失敗しました: {str(e)}")

@app.get("/chat/history/{session_id}")
async def get_chat_history_by_session(session_id: str, limit: int = 50):
    """セッション別のチャット履歴を取得"""
    try:
        history = db_manager.get_chat_history(session_id, limit)
        return {
            "status": "success",
            "session_id": session_id,
            "count": len(history),
            "history": history
        }
    except Exception as e:
        logger.error(f"チャット履歴取得エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=f"チャット履歴の取得に失敗しました: {str(e)}")

@app.get("/chat/history")
async def get_all_chat_history(limit: int = 100):
    """全チャット履歴を取得"""
    try:
        history = db_manager.get_all_chat_history(limit)
        return {
            "status": "success",
            "count": len(history),
            "history": history
        }
    except Exception as e:
        logger.error(f"チャット履歴取得エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=f"チャット履歴の取得に失敗しました: {str(e)}")

@app.delete("/chat/history/{session_id}")
async def delete_chat_history_by_session(session_id: str):
    """セッション別のチャット履歴を削除"""
    try:
        db_manager.delete_chat_history(session_id)
        return {
            "status": "success",
            "message": f"セッション {session_id} の履歴を削除しました"
        }
    except Exception as e:
        logger.error(f"チャット履歴削除エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=f"チャット履歴の削除に失敗しました: {str(e)}")

@app.get("/rag/searches")
async def get_rag_searches(limit: int = 50):
    """RAG検索履歴を取得"""
    try:
        searches = db_manager.get_rag_searches(limit)
        return {
            "status": "success",
            "count": len(searches),
            "searches": searches
        }
    except Exception as e:
        logger.error(f"RAG検索履歴取得エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=f"RAG検索履歴の取得に失敗しました: {str(e)}")

@app.get("/rag/popular-queries")
async def get_popular_queries(limit: int = 10):
    """人気の検索クエリを取得"""
    try:
        queries = db_manager.get_popular_queries(limit)
        return {
            "status": "success",
            "count": len(queries),
            "queries": queries
        }
    except Exception as e:
        logger.error(f"人気クエリ取得エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=f"人気クエリの取得に失敗しました: {str(e)}")

@app.get("/statistics")
async def get_statistics():
    """データベース統計情報を取得"""
    try:
        stats = db_manager.get_statistics()
        return {
            "status": "success",
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"統計情報取得エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=f"統計情報の取得に失敗しました: {str(e)}")

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    documents = rag_manager.get_documents()
    stats = db_manager.get_statistics()
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "loaded_models": list(model_cache.keys()),
        "current_model": current_model_id,
        "rag_enabled": True,
        "documents_count": len(documents),
        "database_stats": stats
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
