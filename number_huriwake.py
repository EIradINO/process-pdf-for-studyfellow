import os
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore
import random

load_dotenv()

# Firebaseの初期化
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

def assign_random_numbers():
    try:
        # statusが'unprocessed'のドキュメントを取得
        unprocessed_docs = db.collection('document_metadata').where('status', '==', 'unprocessed').stream()
        
        for doc in unprocessed_docs:
            # 1から6までのランダムな数字を生成
            random_number = random.randint(1, 6)
            
            # ドキュメントを更新
            db.collection('document_metadata').document(doc.id).update({
                'random': random_number
            })
            print(f"ドキュメントID {doc.id} にランダムな数字 {random_number} を割り当てました")
            
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        return None

if __name__ == "__main__":
    assign_random_numbers()
