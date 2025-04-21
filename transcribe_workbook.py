import os
import sys
import json
from dotenv import load_dotenv
from supabase import create_client, Client
from PyPDF2 import PdfReader, PdfWriter
import io
import google.generativeai as genai
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

load_dotenv()

class PhysicsAnalysis(BaseModel):
    problem_summary: Dict[str, str] = Field(
        default_factory=lambda: {
            "physical_phenomenon": "",
            "main_components": ""
        },
        description="問題の要約",
        json_schema_extra={
            "type": "object",
            "properties": {
                "physical_phenomenon": {"type": "string"},
                "main_components": {"type": "string"}
            }
        }
    )
    main_physics_field: Dict[str, str] = Field(
        default_factory=lambda: {
            "field": "",
            "subfield": "",
            "field_fusion": ""
        },
        description="主たる物理分野",
        json_schema_extra={
            "type": "object",
            "properties": {
                "field": {"type": "string"},
                "subfield": {"type": "string"},
                "field_fusion": {"type": "string"}
            }
        }
    )
    problem_structure: Dict[str, Any] = Field(
        default_factory=lambda: {
            "question_structure": "",
            "diagrams_graphs": "",
            "answer_format": ""
        },
        description="問題形式と構成",
        json_schema_extra={
            "type": "object",
            "properties": {
                "question_structure": {"type": "string"},
                "diagrams_graphs": {"type": "string"},
                "answer_format": {"type": "string"}
            }
        }
    )
    required_abilities: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "knowledge_application": [],
            "physical_laws_application": [],
            "mathematical_processing": [],
            "reading_comprehension": [],
            "modeling": [],
            "experimental_interpretation": []
        },
        description="問われている中心的能力",
        json_schema_extra={
            "type": "object",
            "properties": {
                "knowledge_application": {"type": "array", "items": {"type": "string"}},
                "physical_laws_application": {"type": "array", "items": {"type": "string"}},
                "mathematical_processing": {"type": "array", "items": {"type": "string"}},
                "reading_comprehension": {"type": "array", "items": {"type": "string"}},
                "modeling": {"type": "array", "items": {"type": "string"}},
                "experimental_interpretation": {"type": "array", "items": {"type": "string"}}
            }
        }
    )
    key_laws_formulas: List[str] = Field(
        default_factory=list,
        description="解答に必要な主要法則・公式",
        json_schema_extra={
            "type": "array",
            "items": {"type": "string"}
        }
    )
    mathematical_elements: Dict[str, Any] = Field(
        default_factory=lambda: {
            "math_level": "",
            "important_techniques": [],
            "calculation_complexity": ""
        },
        description="数学的要素の分析",
        json_schema_extra={
            "type": "object",
            "properties": {
                "math_level": {"type": "string"},
                "important_techniques": {"type": "array", "items": {"type": "string"}},
                "calculation_complexity": {"type": "string"}
            }
        }
    )
    difficulty_assessment: Dict[str, Any] = Field(
        default_factory=lambda: {
            "overall_level": "",
            "difficulty_factors": [],
            "problem_type": ""
        },
        description="難易度評価",
        json_schema_extra={
            "type": "object",
            "properties": {
                "overall_level": {"type": "string"},
                "difficulty_factors": {"type": "array", "items": {"type": "string"}},
                "problem_type": {"type": "string"}
            }
        }
    )
    features_and_notes: Dict[str, Any] = Field(
        default_factory=lambda: {
            "guidance_level": "",
            "approximation_requirements": [],
            "novelty": "",
            "key_points": [],
            "traps": []
        },
        description="問題の特徴と注意点",
        json_schema_extra={
            "type": "object",
            "properties": {
                "guidance_level": {"type": "string"},
                "approximation_requirements": {"type": "array", "items": {"type": "string"}},
                "novelty": {"type": "string"},
                "key_points": {"type": "array", "items": {"type": "string"}},
                "traps": {"type": "array", "items": {"type": "string"}}
            }
        }
    )

def transcribe_pdf(pdf_data, problem_number: int) -> tuple[str, str]:
    """PDFデータを文字起こし（問題文と解答を分けて）"""
    try:
        # Gemini APIの初期化
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        model = genai.GenerativeModel(
            'gemini-2.5-flash-preview-04-17',
            system_instruction='''
            あなたは正確な文字起こしAIです。与えられたPDFを正確に文字起こししてください。
            数式はTex形式で出力してください。
            pdf内にグラフや図表があった場合、グラフや図表の説明を文字起こしした文章の適切な位置に挿入してください。
            ページ番号やヘッダーフッターに書かれた共通の章タイトルを含めることを禁止します。
            回答は全て日本語で行なってください。
            '''
        )
        
        # 文字起こしの実行
        prompt = '''
        与えられたPDFから問題文と解答を正確に文字起こししてください。
        問題文と解答は分けて出力してください。
        '''
        
        # 生成設定の定義
        generation_config = {
            'response_mime_type': 'application/json',
            'response_schema': {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"}
                },
                "required": ["question", "answer"]
            }
        }
        
        # コンテンツの生成
        response = model.generate_content(
            contents=[
                prompt,
                {"mime_type": "application/pdf", "data": pdf_data}
            ],
            generation_config=generation_config
        )
        
        # レスポンスの検証
        if not response.text:
            print(f"問題 {problem_number} の処理がスキップされました")
            return "", ""
        
        # レスポンスをそのまま出力
        print(f"\n問題 {problem_number} の生のレスポンス:")
        print(response.text)
        print("\n" + "="*50 + "\n")
        
        # テキストをJSONに変換
        json_data = json.loads(response.text)
        return json_data["question"], json_data["answer"]
        
    except Exception as e:
        print(f"問題 {problem_number} の文字起こし中にエラーが発生しました: {str(e)}")
        return "", ""

def analyze_problem(question: str, answer: str) -> str:
    """問題文と解答を分析して解説を生成"""
    try:
        # Gemini APIの初期化
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        model = genai.GenerativeModel(
            'gemini-2.5-pro-preview-03-25',
            system_instruction='''
            あなたは優秀な物理学者です。与えられた問題文と解答を以下の指標に従って詳細に分析してください。

            問題設定の要約:
                扱っている物理現象・状況の簡単な説明（例：単振り子の運動、RLC回路の過渡現象、気体の状態変化と熱効率）
                主要な構成要素（物体、場、装置など）
            主たる物理分野:
                該当する分野を特定（力学、熱力学、波動、電磁気学、原子物理）
                （該当する場合）分野内の詳細テーマ（例：力学→円運動、電磁気学→コンデンサー）
                （該当する場合）複数分野の融合度（どの分野がどのように関連しているか）
            問題形式と構成:
                大問・小問の構成（設問数、独立性、連続性）
                図・グラフの有無とその役割（状況理解補助、データ提示、解答の一部）
                想定される解答形式（選択、数値記入、記号選択、記述説明、途中式記述、グラフ描画）
            問われている中心的能力:
                知識・公式の理解と適用
                物理法則の応用・深い考察
                数学的処理能力（計算、近似、ベクトル、微積分）
                読解力・情報整理能力
                モデル化・仮定の設定能力
                実験・観察データの解釈・考察能力
            解答に必要な主要法則・公式:
                問題解決に不可欠な物理法則、原理、公式を列挙（例：運動量保存則、エネルギー保存則、キルヒホッフの法則、熱力学第一法則、光の干渉条件）
            数学的要素の分析:
                要求される数学レベル（例：数I・A、数II・B、数III）
                特に重要な数学的手法（例：三角関数、ベクトル、微分、積分、近似計算）
                計算量の評価（少ない、標準的、多い、複雑）
            難易度評価:
                総合的な難易度レベル（基礎、標準、応用、難関）
                難易度を構成する要因（設定の複雑さ、思考ステップ数、計算量、見慣れない題材、時間制限）
                問題の典型度（典型問題、標準的な応用問題、思考力重視の独自問題）
            問題の特徴と注意点:
                誘導の丁寧さ（丁寧なステップ、ヒント少なめ、自力での思考要求）
                近似計算の要否とその種類（例：微小角近似 sinθ≈θ, (1+x)^n≈1+nx）
                設定の新規性・独創性
                解法のポイント、注意すべき物理的・数学的トラップ

            回答は全て日本語で行なってください。
            '''
        )
        
        # 分析の実行
        prompt = f'''
        問題文:
        {question}

        解答:
        {answer}
        '''
        
        response = model.generate_content(prompt)
        print(response.text)
        if not response.text:
            print("問題の分析がスキップされました")
            return ""
        
        return response.text
        
    except Exception as e:
        print(f"問題の分析中にエラーが発生しました: {str(e)}")
        return ""

def structure_analysis(analysis: str) -> Dict[str, Any]:
    """問題文、解答、解説を構造化された分析データに変換"""
    try:
        # Gemini APIの初期化
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        model = genai.GenerativeModel(
            'gemini-1.5-flash',
            system_instruction='''
            あなたは正確なjson変換マシーンです。与えられた分析結果を、jsonデータに変換して出力してください。
            '''
        )
        
        # 生成設定の定義
        generation_config = {
            'response_mime_type': 'application/json',
            'response_schema': PhysicsAnalysis,
        }
        
        # コンテンツの生成
        response = model.generate_content(
            contents=[analysis],
            generation_config=generation_config
        )
        
        # レスポンスの検証
        if not response.text:
            print("構造化分析がスキップされました")
            return {}
        
        # レスポンスをそのまま出力
        print("\n構造化分析の生のレスポンス:")
        print(response.text)
        print("\n" + "="*50 + "\n")
        
        # テキストをJSONに変換してPhysicsAnalysisに変換
        json_data = json.loads(response.text)
        result = PhysicsAnalysis(**json_data)
        
        # 結果を辞書形式で返す
        return result.model_dump()
        
    except Exception as e:
        print(f"構造化分析中にエラーが発生しました: {str(e)}")
        return {}

def save_to_supabase(supabase: Client, question: str, answer: str, analysis: str, structured_analysis: dict, problem_number: int, file_name: str, document_id: str):
    """文字起こし結果をSupabaseに保存"""
    try:
        data = {
            "question": question,
            "answer": answer,
            "analysis": analysis,
            "structured_analysis": structured_analysis,
            "problem_number": problem_number,
            "file_name": file_name,
            "document_id": document_id
        }
        
        response = supabase.table('workbook_transcriptions').insert(data).execute()
        print(f"問題 {problem_number} の結果を保存しました")
        return response.data
    except Exception as e:
        print(f"保存中にエラーが発生しました: {str(e)}")
        return None

def process_workbook(problems_file, target_file_name):
    try:
        # 問題情報のJSONファイルを読み込み
        with open(problems_file, 'r') as f:
            problems = json.load(f)
        
        # Supabaseクライアントの初期化
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # 指定されたファイル名のドキュメントを取得
        response = supabase.table('document_metadata').select("*").eq('file_name', target_file_name).execute()
        
        if not response.data:
            print(f"ファイル {target_file_name} が見つかりませんでした。")
            return None
            
        doc = response.data[0]
        file_name = doc.get('file_name')
        print(f"\nファイル {file_name} の処理を開始します...")
        
        try:
            # Storageからファイルをダウンロード
            pdf_data = supabase.storage.from_('workbooks').download(file_name)
            print(f"ファイル {file_name} のダウンロードが完了しました")
            
            # PDFリーダーの初期化
            reader = PdfReader(io.BytesIO(pdf_data))
            
            # 各問題を順次処理
            for problem in problems:
                print(f"\n問題 {problem['problem_number']} の処理を開始します...")
                
                try:
                    # 問題ごとにPDFを分割
                    writer = PdfWriter()
                    start_page = problem['start_page']
                    end_page = problem['end_page']
                    
                    # 指定されたページ範囲のページを追加
                    for page_num in range(start_page - 1, end_page):
                        if page_num < len(reader.pages):
                            writer.add_page(reader.pages[page_num])
                    
                    # メモリ上でPDFを保持
                    buffer = io.BytesIO()
                    writer.write(buffer)
                    pdf_chunk = buffer.getvalue()
                    
                    # 文字起こしの実行
                    try:
                        question, answer = transcribe_pdf(pdf_chunk, problem['problem_number'])
                    except Exception as e:
                        print(f"文字起こし中にエラーが発生しました: {str(e)}")
                        question, answer = "", ""
                    
                    # 問題の分析を実行
                    try:
                        analysis = analyze_problem(question, answer)
                    except Exception as e:
                        print(f"問題分析中にエラーが発生しました: {str(e)}")
                        analysis = ""
                    
                    # 構造化分析を実行
                    try:
                        structured_analysis = structure_analysis(analysis)
                    except Exception as e:
                        print(f"構造化分析中にエラーが発生しました: {str(e)}")
                        structured_analysis = {}
                    
                    # 結果を保存（エラーが発生しても空のデータとして保存）
                    save_to_supabase(
                        supabase,
                        question,
                        answer,
                        analysis,
                        structured_analysis,
                        problem['problem_number'],
                        file_name,
                        doc['id']
                    )
                    
                    # メモリの解放
                    del writer
                    del buffer
                    del pdf_chunk
                
                except Exception as e:
                    print(f"問題 {problem['problem_number']} の処理中にエラーが発生しました: {str(e)}")
                    # エラーが発生しても空のデータとして保存
                    save_to_supabase(
                        supabase,
                        "",
                        "",
                        "",
                        {},
                        problem['problem_number'],
                        file_name,
                        doc['id']
                    )
            
            print(f"ファイル {file_name} の処理が完了しました")
            
        except Exception as e:
            print(f"ファイル {file_name} の処理中にエラーが発生しました: {str(e)}")
    
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        return None

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("使用方法: python3 transcribe_workbook.py target_file_name problem_numbers/sample.json")
        sys.exit(1)
    
    target_file_name = sys.argv[1]
    problems_file = sys.argv[2]
    process_workbook(problems_file, target_file_name)
