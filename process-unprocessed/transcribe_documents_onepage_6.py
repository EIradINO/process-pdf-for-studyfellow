import os
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore, storage
from PyPDF2 import PdfReader, PdfWriter
import io
from google import genai
from google.genai import types
import concurrent.futures
from typing import List, Dict
import asyncio

load_dotenv()

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    "storageBucket": "studyfellow-42d35.firebasestorage.app"  # ← ご自身のバケット名に変更
})
db = firestore.client()
bucket = storage.bucket()

def split_pdf_in_memory(pdf_data):
    """PDFを1ページずつ分割してメモリ上で保持"""
    reader = PdfReader(io.BytesIO(pdf_data))
    total_pages = len(reader.pages)
    split_pdfs = []
    
    for i in range(total_pages):
        writer = PdfWriter()
        writer.add_page(reader.pages[i])
        
        buffer = io.BytesIO()
        writer.write(buffer)
        split_pdfs.append({
            "name": f"page_{i+1}",
            "data": buffer.getvalue(),
            "page": i + 1
        })
        print(f"ページ {i+1} の分割が完了しました")
    
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
            return ""
        
        print(f"\nページ {page} の処理結果:")
        print(response.text)
        
        return response.text
        
    except Exception as e:
        print(f"ページ {page} の文字起こし中にエラーが発生しました: {str(e)}")
        return ""

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
            if transcription:
                save_to_firestore(transcription, pdf['page'], file_name, doc_id)

def process_documents():
    try:
        # 削除されたドキュメントの処理
        deleted_docs = db.collection('document_metadata').where('status', '==', 'deleted').stream()
        deleted_docs = [doc for doc in deleted_docs]
        if deleted_docs:
            for doc in deleted_docs:
                doc_id = doc.id
                # document_transcriptionsの削除
                trans_docs = db.collection('document_transcriptions').where('document_id', '==', doc_id).stream()
                for tdoc in trans_docs:
                    db.collection('document_transcriptions').document(tdoc.id).delete()
                print(f"ドキュメントID {doc_id} の文字起こしデータを削除しました")
                # ステータス更新
                db.collection('document_metadata').document(doc_id).update({'status': 'deleted_applied'})
                print(f"ドキュメントID {doc_id} のステータスをdeleted_appliedに更新しました")
        
        # 1. statusが'unprocessed'のdocument_metadataのidを取得
        unprocessed_docs = db.collection('document_metadata').where('status', '==', 'unprocessed').stream()
        unprocessed_ids = [doc.id for doc in unprocessed_docs]
        # 2. それらのidをdocument_idに持つdocument_transcriptionsを削除
        for doc_id in unprocessed_ids:
            trans_docs = db.collection('document_transcriptions').where('document_id', '==', doc_id).stream()
            for tdoc in trans_docs:
                db.collection('document_transcriptions').document(tdoc.id).delete()
        print(f"unprocessedなdocument_metadataに紐づくdocument_transcriptionsを全削除しました")
        
        # 未処理ドキュメントの取得
        response = db.collection('document_metadata').where('status', '==', 'unprocessed').where('random', '==', 6).stream()
        docs = [doc.to_dict() | {'id': doc.id} for doc in response]
        if not docs:
            print("処理待ちのドキュメントが見つかりませんでした。")
            return None
        
        for doc in docs:
            file_name = doc.get('file_name')
            doc_id = doc.get('id')
            file_path = doc.get('path')  # Firestoreのpathフィールドを利用
            print(f"\nファイル {file_name} の処理を開始します...")
            try:
                blob = bucket.blob(file_path)
                pdf_data = blob.download_as_bytes()
                print(f"ファイル {file_name} のダウンロードが完了しました")
                split_pdfs = split_pdf_in_memory(pdf_data)
                print(f"ファイル {file_name} の分割が完了しました")
                # 100ページずつバッチ処理
                batch_size = 100
                for i in range(0, len(split_pdfs), batch_size):
                    batch = split_pdfs[i:i + batch_size]
                    print(f"\nページ {i+1} から {i+len(batch)} のバッチ処理を開始します...")
                    asyncio.run(process_page_batch(batch, file_name, doc_id))
                db.collection('document_metadata').document(doc_id).update({'status': 'processed'})
                print(f"ファイル {file_name} の処理が完了しました")
            except Exception as e:
                print(f"ファイル {file_name} の処理中にエラーが発生しました: {str(e)}")
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        return None

if __name__ == "__main__":
    process_documents()
