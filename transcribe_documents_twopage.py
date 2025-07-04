import os
from dotenv import load_dotenv
from supabase import create_client, Client
from PyPDF2 import PdfReader, PdfWriter
import io
import google.generativeai as genai

load_dotenv()

def split_pdf_in_memory(pdf_data):
    """PDFを表紙と2ページずつに分割してメモリ上で保持"""
    reader = PdfReader(io.BytesIO(pdf_data))
    total_pages = len(reader.pages)
    split_pdfs = []
    
    cover_writer = PdfWriter()
    cover_writer.add_page(reader.pages[0])
    cover_buffer = io.BytesIO()
    cover_writer.write(cover_buffer)
    split_pdfs.append({
        "name": "cover",
        "data": cover_buffer.getvalue(),
        "page": 1
    })
    print("表紙の分割が完了しました")
    
    for i in range(1, total_pages, 2):
        writer = PdfWriter()
        writer.add_page(reader.pages[i])
        if i + 1 < total_pages:
            writer.add_page(reader.pages[i + 1])
        
        buffer = io.BytesIO()
        writer.write(buffer)
        split_pdfs.append({
            "name": f"pages_{i+1}",
            "data": buffer.getvalue(),
            "page": i + 1
        })
        print(f"ページ {i+1} の分割が完了しました")
    
    return split_pdfs

def transcribe_pdf(pdf_data, page: int):
    """PDFデータを文字起こし"""
    try:
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash-preview-04-17",
            system_instruction="""
            あなたは柔軟性を持つ文字起こしAIです。ユーザーの指示に従ってPDFをMarkdown形式で文字起こしして結果のみ表示してください。
            不自然に文の途中で改行することを禁止します。
            """
        )
        
        prompt = """
        数式はTeX形式で記述してください。
        pdf内にグラフや図表があった場合、グラフや図表の説明を文字起こしした文章の適切な位置に挿入してください。
        ページ番号やヘッダーフッターに書かれた共通の章タイトルを含めることを禁止します。
        """
        
        response = model.generate_content(
            [prompt, {"mime_type": "application/pdf", "data": pdf_data}]
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

def save_to_supabase(supabase: Client, transcription: str, page: int, file_name: str, document_id: str):
    """文字起こし結果をSupabaseに保存"""
    try:
        data = {
            "transcription": transcription,
            "page": page,
            "file_name": file_name,
            "document_id": document_id
        }
        
        response = supabase.table('document_transcriptions').insert(data).execute()
        print(f"ページ {page} の結果を保存しました")
        return response.data
    except Exception as e:
        print(f"保存中にエラーが発生しました: {str(e)}")
        return None

def process_documents():
    try:
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        supabase: Client = create_client(supabase_url, supabase_key)
        
        deleted_docs = supabase.table('document_metadata').select("id").eq('status', 'deleted').execute()
        
        if deleted_docs.data:
            for doc in deleted_docs.data:
                doc_id = doc.get('id')
                supabase.table('document_transcriptions').delete().eq('document_id', doc_id).execute()
                print(f"ドキュメントID {doc_id} の文字起こしデータを削除しました")
                
                supabase.table('document_metadata').update({'status': 'deleted_applied'}).eq('id', doc_id).execute()
                print(f"ドキュメントID {doc_id} のステータスをdeleted_appliedに更新しました")
        
        response = supabase.table('document_metadata').select("*").eq('status', 'unprocessed').execute()
        
        if not response.data:
            print("処理待ちのドキュメントが見つかりませんでした。")
            return None
            
        for doc in response.data:
            file_name = doc.get('file_name')
            doc_id = doc.get('id')
            bucket = doc.get('bucket')
            print(f"\nファイル {file_name} の処理を開始します...")
            
            try:
                pdf_data = supabase.storage.from_(bucket).download(file_name)
                print(f"ファイル {file_name} のダウンロードが完了しました")
                
                split_pdfs = split_pdf_in_memory(pdf_data)
                print(f"ファイル {file_name} の分割が完了しました")
                
                for pdf in split_pdfs:
                    print(f"\n{pdf['name']} の文字起こしを開始します...")
                    transcription = transcribe_pdf(pdf['data'], pdf['page'])
                    save_to_supabase(supabase, transcription, pdf['page'], file_name, doc_id)
                
                supabase.table('document_metadata').update({'status': 'processed'}).eq('file_name', file_name).execute()
                print(f"ファイル {file_name} の処理が完了しました")
                
            except Exception as e:
                print(f"ファイル {file_name} の処理中にエラーが発生しました: {str(e)}")
        
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        return None

if __name__ == "__main__":
    process_documents()
