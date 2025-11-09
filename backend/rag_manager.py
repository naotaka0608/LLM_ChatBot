"""
RAG (Retrieval-Augmented Generation) マネージャー
ドキュメントの読み込み、埋め込み生成、検索機能を提供
"""

import os
from typing import List, Dict, Optional
from pathlib import Path
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from PyPDF2 import PdfReader

logger = logging.getLogger(__name__)

class RAGManager:
    """RAG機能を管理するクラス"""

    def __init__(self, upload_dir: str = "./uploads", persist_dir: str = "./vector_store"):
        """
        初期化

        Args:
            upload_dir: アップロードファイルの保存ディレクトリ
            persist_dir: ベクトルストアの永続化ディレクトリ
        """
        self.upload_dir = Path(upload_dir)
        self.persist_dir = Path(persist_dir)
        self.upload_dir.mkdir(exist_ok=True)
        self.persist_dir.mkdir(exist_ok=True)

        # 埋め込みモデル（日本語対応）
        logger.info("埋め込みモデルをロード中...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        # ベクトルストア
        self.vector_store: Optional[FAISS] = None
        self.documents: List[Dict] = []

        # テキスト分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )

        # 既存のベクトルストアをロード
        self._load_vector_store()

        logger.info("RAGマネージャーの初期化完了")

    def _load_vector_store(self):
        """永続化されたベクトルストアをロード"""
        index_path = self.persist_dir / "index.faiss"
        if index_path.exists():
            try:
                self.vector_store = FAISS.load_local(
                    str(self.persist_dir),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("既存のベクトルストアをロードしました")
            except Exception as e:
                logger.warning(f"ベクトルストアのロードに失敗: {e}")
                self.vector_store = None

    def _save_vector_store(self):
        """ベクトルストアを永続化"""
        if self.vector_store:
            try:
                self.vector_store.save_local(str(self.persist_dir))
                logger.info("ベクトルストアを保存しました")
            except Exception as e:
                logger.error(f"ベクトルストアの保存に失敗: {e}")

    def read_pdf(self, file_path: Path) -> str:
        """PDFファイルを読み込む"""
        try:
            reader = PdfReader(str(file_path))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"PDFの読み込みエラー: {e}")
            raise

    def read_text_file(self, file_path: Path) -> str:
        """テキストファイル（txt, md）を読み込む"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # UTF-8で失敗した場合、Shift-JISで試す
            with open(file_path, 'r', encoding='shift-jis') as f:
                return f.read()

    def process_document(self, file_path: Path, filename: str) -> Dict:
        """
        ドキュメントを処理してベクトルストアに追加

        Args:
            file_path: ファイルパス
            filename: ファイル名

        Returns:
            処理結果の辞書
        """
        logger.info(f"ドキュメント処理開始: {filename}")

        # ファイル拡張子に応じて読み込み
        suffix = file_path.suffix.lower()
        if suffix == '.pdf':
            text = self.read_pdf(file_path)
        elif suffix in ['.txt', '.md']:
            text = self.read_text_file(file_path)
        else:
            raise ValueError(f"未対応のファイル形式: {suffix}")

        # テキストを分割
        chunks = self.text_splitter.split_text(text)
        logger.info(f"{len(chunks)}個のチャンクに分割しました")

        # メタデータを付与
        metadatas = [{"source": filename, "chunk": i} for i in range(len(chunks))]

        # ベクトルストアに追加
        if self.vector_store is None:
            # 初回作成
            self.vector_store = FAISS.from_texts(
                chunks,
                self.embeddings,
                metadatas=metadatas
            )
        else:
            # 既存のストアに追加
            new_store = FAISS.from_texts(
                chunks,
                self.embeddings,
                metadatas=metadatas
            )
            self.vector_store.merge_from(new_store)

        # 永続化
        self._save_vector_store()

        # ドキュメント情報を保存
        doc_info = {
            "filename": filename,
            "file_path": str(file_path),
            "chunks": len(chunks),
            "characters": len(text)
        }
        self.documents.append(doc_info)

        logger.info(f"ドキュメント処理完了: {filename}")
        return doc_info

    def search(self, query: str, k: int = 3) -> List[Dict]:
        """
        クエリに関連するドキュメントを検索

        Args:
            query: 検索クエリ
            k: 返す結果の数

        Returns:
            検索結果のリスト
        """
        if self.vector_store is None:
            logger.warning("ベクトルストアが空です")
            return []

        try:
            # 類似度検索
            docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k)

            results = []
            for doc, score in docs_with_scores:
                results.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "chunk": doc.metadata.get("chunk", 0),
                    "score": float(score)
                })

            return results
        except Exception as e:
            logger.error(f"検索エラー: {e}")
            return []

    def get_documents(self) -> List[Dict]:
        """アップロードされたドキュメントのリストを取得"""
        return self.documents

    def delete_all_documents(self):
        """全てのドキュメントを削除"""
        self.vector_store = None
        self.documents = []

        # 永続化ファイルを削除
        index_path = self.persist_dir / "index.faiss"
        pkl_path = self.persist_dir / "index.pkl"

        if index_path.exists():
            index_path.unlink()
        if pkl_path.exists():
            pkl_path.unlink()

        logger.info("全てのドキュメントを削除しました")
