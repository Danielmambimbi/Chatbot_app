import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import random
from flask import Flask, request,render_template,jsonify
from flask_cors import CORS
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Ajoutez en tête de fichier
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Essentiel pour Railway

file=os.path.dirname(os.path.abspath(__file__))
# 🔹 Charger le fichier CSV
df = pd.read_csv(file + "/faq_222.csv",encoding='ISO-8859-1', sep=';')  # Assure-toi que ce fichier est dans le même dossier

# 🔹 Séparer les données
X = df["reponse"]
y = df["intent"]


det_intent_quest=joblib.load(file+"/Models/det_intent_quest.pkl")


det_intent_res=joblib.load(file+"/Models/det_intent_res.pkl")

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Charger la FAQ
file=os.path.dirname(os.path.abspath(__file__))
# 🔹 Charger le fichier CSV
df = pd.read_csv(file + "/faq_222.csv",encoding='ISO-8859-1', sep=';')  # Assure-toi que ce fichier est dans le même dossier

questions = df['question'].tolist()
reponses = df['reponse'].tolist()

intents_responses={}
t_total_intent=len(y)
i=0
while True:
    if y[i] in intents_responses:
        intents_responses[y[i]].append(X[i])
    else:
        intents_responses[y[i]]=[]
        intents_responses[y[i]].append(X[i])
    i+=1
    if i==t_total_intent:
        break



# print("questions : ",len(questions))
# print("reponses : ",len(reponses))
# Encoder toutes les questions une seule fois
question_embeddings = np.load(file + "/Models/question_embeddings.npy")

res_error=["Je suis désolé, j'ai ne pas bien compris votre demande. Pouvez-vous reformuler, s'il vous plaît ?",
           "Oups, je n'ai pas tout saisi. Pourriez-vous préciser ce que vous souhaitze faire ?",
           "Je ne suis pas sûr de comprendre. Pourriez-vous me donner un peu plus de détails ?",
           "Je n’ai pas bien compris votre question. Est-ce que cela concerne une commande, un retour ou un produit en particulier ?",
           "Pouvez-vous m’en dire un peu plus pour que je puisse mieux vous aider ?",
           "Est-ce que votre question concerne une commande passée, un produit, ou un problème technique ?",
           "Pour que je puisse vous aider plus efficacement, pouvez-vous préciser si vous parlez d’un échange, d’un remboursement ou d’un problème de livraison ?",
           "Je n’ai pas bien compris votre message. Pourriez-vous indiquer votre numéro de commande si cela concerne un achat ?",
           "Je n’ai pas bien compris votre demande. Je vais transférer votre message à un conseiller qui pourra vous aider.",
           "Je comprends que ce soit frustrant. Je veux vraiment vous aider, pouvez-vous reformuler votre question ?",
           "Merci pour votre message. Il semble un peu flou de mon côté. Pourriez-vous me donner plus de précisions ?",
           "Je suis là pour vous aider, mais j’ai besoin d’un peu plus d’informations pour comprendre votre demande.",
           "Je n’ai pas trouvé d’information correspondant à votre demande. Pouvez-vous reformuler en quelques mots simples ?",
           "Votre message ne correspond pas à ce que je peux traiter automatiquement. Essayons ensemble : parlez-vous d’une commande, d’un produit ou d’un retour ?",
           "Hmm… Je n’ai pas bien compris 🧐 Pouvez-vous réessayer avec d’autres mots ?",
           "Je ne suis pas sûr d’avoir bien saisi 🤖 Vous pourriez reformuler ou choisir une option ici pour m’aider à mieux vous guider."
           ]

def seuil_response(question_utilisateur, seuil=0.65):   
    response={}
    vecteur = model.encode([question_utilisateur])
    similarites = cosine_similarity(vecteur, question_embeddings)[0]
    index_max = np.argmax(similarites)
    score = similarites[index_max]
    response={
        "score":score,
        "index_max":index_max
    }
    return response


def chatbot(user_input):
    resp=seuil_response(user_input)
    input_error="no error"
    score=resp["score"]
    index_max=resp["index_max"]
    pred_intent_ques=det_intent_quest.predict([user_input])[0]
    pred_intent_res=det_intent_res.predict([reponses[index_max]])[0]
    res_chat=reponses[index_max]    
    
    if (score>=0.60):
        if (pred_intent_ques==pred_intent_res):
            res_chat=reponses[index_max]
        else:
           res_chat=intents_responses[pred_intent_ques]
           res_chat=random.choice(res_chat)
        input_error="no error"
    else:
        res_chat=random.choice(res_error)
        input_error="error"
    
    response={
        "input_error":input_error,
        "res_chat":res_chat,
        "score":score,
        "det_intent_quest":pred_intent_ques,
        "det_intent_res":pred_intent_res
    }
    
    return response


app=Flask(__name__)
CORS(app)
@app.errorhandler(404)
def page_not_found(error):
    return 'This page does not exist', 404
@app.route('/')
def home():
    return "Hello word !"
@app.route('/health')
def health_check():
    return jsonify({"status": "healthy"}), 200
@app.route('/chat',methods=["POST"])
def chat():
    att="Veuillez patienter"
    response=""
    user_input=request.form["message"]
    response=chatbot(user_input)
    score=response["score"]
    intent_quest=response["det_intent_quest"]
    intent_res=response["det_intent_res"]
    input_error=response["input_error"]
    response=response["res_chat"]
    
    file=os.path.dirname(os.path.abspath(__file__))
    fichier=file+"/promts.txt"
    ajout=f"promt:{user_input}\t intent_quest={intent_quest} \t intent_res={intent_res}\t scrore={score}\n"

    with open(fichier,"a",encoding="utf-8") as f:
        f.write(ajout)
    
    all_response={
                "response_trans":response,
                "att":att,
                "input_error":input_error
            }
    response=all_response
    return response
# [...] (tout le reste de votre code)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)


