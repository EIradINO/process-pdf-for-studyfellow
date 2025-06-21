import os
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore, storage
from PyPDF2 import PdfReader, PdfWriter
import io
from google import genai
from google.genai import types
import concurrent.futures
from typing import List, Dict, Set
import asyncio

load_dotenv()

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    "storageBucket": "studyfellow-42d35.firebasestorage.app"  # ← ご自身のバケット名に変更
})
db = firestore.client()
bucket = storage.bucket()

def extract_specific_pages_from_pdf(pdf_data, missing_pages: List[int]):
    """PDFから指定されたページのみを抽出してメモリ上で保持"""
    reader = PdfReader(io.BytesIO(pdf_data))
    split_pdfs = []
    
    for page_num in missing_pages:
        if page_num <= len(reader.pages):
            writer = PdfWriter()
            writer.add_page(reader.pages[page_num - 1])  # ページ番号は1ベースだが、インデックスは0ベース
            
            buffer = io.BytesIO()
            writer.write(buffer)
            split_pdfs.append({
                "name": f"page_{page_num}",
                "data": buffer.getvalue(),
                "page": page_num
            })
            print(f"ページ {page_num} の抽出が完了しました")
    
    return split_pdfs

def transcribe_pdf(pdf_data, page: int):
    """PDFデータを文字起こし"""
    try:
        client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY_6"))
        
        prompt = """
        数式はTeX形式で記述してください。
        pdf内にグラフや図表があった場合、グラフや図表の説明を文字起こしした文章の適切な位置に挿入してください。
        ページ番号やヘッダーフッターに書かれた共通の章タイトルを含めることを禁止します。
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-05-20",
            config=types.GenerateContentConfig(
                system_instruction="あなたは柔軟性を持つ文字起こしAIです。ユーザーの指示に従ってPDFをMarkdown形式で文字起こしして結果のみ表示してください。不自然に文の途中で改行することを禁止します。"
            ),
            contents=[
                types.Part.from_bytes(
                    data=pdf_data,
                    mime_type='application/pdf'
                ),
                prompt
            ]
        )
        
        if not response.candidates:
            print(f"ページ {page} の処理がスキップされました（著作権の可能性があります）")
            return ""
            
        if not response.text:
            print(f"ページ {page} の処理がスキップされました（レスポンスが空です）")
            return "内容なし"
        
        print(f"\nページ {page} の処理結果:")
        print(response.text)
        
        return response.text
        
    except Exception as e:
        print(f"ページ {page} の文字起こし中にエラーが発生しました: {str(e)}")
        return "内容なし"

def save_to_firestore(transcription: str, page: int, file_name: str, document_id: str):
    """文字起こし結果をFirestoreに保存"""
    try:
        data = {
            "transcription": transcription,
            "page": page,
            "file_name": file_name,
            "document_id": document_id
        }
        db.collection('document_transcriptions').add(data)
        print(f"ページ {page} の結果を保存しました")
    except Exception as e:
        print(f"保存中にエラーが発生しました: {str(e)}")
        return None

async def process_page_batch(pdf_batch: List[Dict], file_name: str, doc_id: str):
    """ページのバッチを並列処理"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for pdf in pdf_batch:
            future = executor.submit(transcribe_pdf, pdf['data'], pdf['page'])
            futures.append((future, pdf))
        
        for future, pdf in futures:
            transcription = future.result()
            if transcription:  # 空文字列以外（「内容なし」も含む）は保存
                save_to_firestore(transcription, pdf['page'], file_name, doc_id)

def get_missing_pages(document_id: str, total_pages: int) -> List[int]:
    """存在しないページ番号を取得"""
    try:
        # 既存の文字起こしページを取得
        trans_docs = db.collection('document_transcriptions').where('document_id', '==', document_id).stream()
        existing_pages: Set[int] = set()
        
        for doc in trans_docs:
            page_num = doc.to_dict().get('page')
            if page_num:
                existing_pages.add(page_num)
        
        # 全ページ(1からtotal_pages)と既存ページの差分を取得
        all_pages = set(range(1, total_pages + 1))
        missing_pages = list(all_pages - existing_pages)
        missing_pages.sort()
        
        print(f"ドキュメントID {document_id}: 総ページ数 {total_pages}, 既存ページ数 {len(existing_pages)}, 不足ページ数 {len(missing_pages)}")
        if missing_pages:
            print(f"不足ページ: {missing_pages}")
        
        return missing_pages
        
    except Exception as e:
        print(f"ページ確認中にエラーが発生しました: {str(e)}")
        return []

def process_error_pages():
    """文字起こしエラーページの再処理"""
    try:
        # statusがprocessedかつrandomが1のドキュメントを取得
        processed_docs = db.collection('document_metadata').where('status', '==', 'processed').where('random', '==', 6).stream()
        docs = [doc.to_dict() | {'id': doc.id} for doc in processed_docs]
        
        if not docs:
            print("再処理対象のドキュメントが見つかりませんでした。")
            return None
        
        print(f"再処理対象のドキュメント数: {len(docs)}")
        
        for doc in docs:
            file_name = doc.get('file_name')
            doc_id = doc.get('id')
            file_path = doc.get('path')
            total_pages = doc.get('total_pages')
            
            if not total_pages:
                print(f"ファイル {file_name}: total_pagesが設定されていないため、スキップします")
                continue
            
            print(f"\nファイル {file_name} の処理を開始します...")
            
            # 不足ページを特定
            missing_pages = get_missing_pages(doc_id, total_pages)
            
            if not missing_pages:
                print(f"ファイル {file_name}: すべてのページが処理済みです")
                continue
            
            try:
                # Firebase Storageからファイルをダウンロード
                blob = bucket.blob(file_path)
                pdf_data = blob.download_as_bytes()
                print(f"ファイル {file_name} のダウンロードが完了しました")
                
                # 不足ページのみを抽出
                split_pdfs = extract_specific_pages_from_pdf(pdf_data, missing_pages)
                print(f"ファイル {file_name} の不足ページ抽出が完了しました")
                
                # 100ページずつバッチ処理
                batch_size = 100
                for i in range(0, len(split_pdfs), batch_size):
                    batch = split_pdfs[i:i + batch_size]
                    batch_pages = [pdf['page'] for pdf in batch]
                    print(f"\nページ {batch_pages} のバッチ処理を開始します...")
                    asyncio.run(process_page_batch(batch, file_name, doc_id))
                
                print(f"ファイル {file_name} の不足ページ処理が完了しました")
                
            except Exception as e:
                print(f"ファイル {file_name} の処理中にエラーが発生しました: {str(e)}")
                
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        return None

if __name__ == "__main__":
    process_error_pages() 