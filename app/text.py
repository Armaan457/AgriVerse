import os
from dotenv import load_dotenv
import numpy as np
from langchain_neo4j import Neo4jGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from PIL import Image
from utils import ANSWER_GEN_TEMPLATE, CYPHER_SUBGRAPH_H, CYPHER_SUBGRAPH_D, cleanup_relationships
from vision import predict_fn, id_to_class
from rapidfuzz import process, fuzz
import numpy as np
load_dotenv()

graph = Neo4jGraph(url="neo4j://127.0.0.1:7687", username=os.getenv("NEO4J_USER"), password=os.getenv("NEO4J_PASSWORD"), enhanced_schema=True)

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

ANSWER_GEN_PROMPT = PromptTemplate.from_template(ANSWER_GEN_TEMPLATE)
answer_chain = ANSWER_GEN_PROMPT | llm

def run_pipeline(query, image=None):

    if image:
        img = Image.open(image).convert("RGB")
        img_np = np.array(img)

        probs = predict_fn([img_np])
        class_id = np.argmax(probs[0])
        predicted_class_name = id_to_class[class_id]

        print(f"[VISION] Predicted: {predicted_class_name}")

        crop, dis = predicted_class_name.split(',')
        print(f"[TEXT] Crop={crop}, Disease={dis}")
        if dis.lower() == "healthy":
            CYPHER_SUBGRAPH = CYPHER_SUBGRAPH_H
            subgraph = graph.query(CYPHER_SUBGRAPH, params={"disease": dis, "crop" : crop })
        else:
            CYPHER_SUBGRAPH = CYPHER_SUBGRAPH_D
            subgraph = graph.query(CYPHER_SUBGRAPH, params={"disease": dis, "crop" : crop })[0]['rels']
            subgraph = cleanup_relationships(subgraph)

        answer = answer_chain.invoke({
            "question": query,
            "context": str(subgraph),
            "crop": crop,
            "disease": dis
        })
        return answer.content, subgraph

    else:
        
        crop, dis = extract_from_text(
            query
        )
        print(f"[DETECTED] Crop={crop}, Disease={dis}")
        if dis.lower() == "healthy":
            CYPHER_SUBGRAPH = CYPHER_SUBGRAPH_H
            subgraph = graph.query(CYPHER_SUBGRAPH, params={"crop" : crop })
        else:
            CYPHER_SUBGRAPH = CYPHER_SUBGRAPH_D
            subgraph = graph.query(CYPHER_SUBGRAPH, params={"disease" : dis})[0]['rels']
            subgraph = cleanup_relationships(subgraph)

        answer = answer_chain.invoke({
            "question": query,
            "context": str(subgraph),
            "crop": crop,
            "disease": dis
        })
        return answer.content, subgraph


def load_crops_and_diseases(images_root='../dataset/images'):
    crops_set = set()
    diseases_set = set()

    if not os.path.isdir(images_root):
        return [], []

    for entry in os.listdir(images_root):
        path = os.path.join(images_root, entry)
        if not os.path.isdir(path):
            continue

        if ',' in entry:
            crop, disease = [p.strip() for p in entry.split(',', 1)]
        else:
            crop = entry.strip()
            disease = ''

        if crop:
            crops_set.add(crop)
        if disease:
            diseases_set.add(disease)

    return sorted(crops_set), sorted(diseases_set)


CROPS, DISEASES = load_crops_and_diseases()


def extract_from_text(query):
    if not query:
        return None, None

    q = query.lower()
    crop = None
    disease = None

    for c in CROPS:
        if c.lower() in q:
            crop = c
            break

    for d in DISEASES:
        if d.lower() in q:
            disease = d
            break

    if not crop:
        match = process.extractOne(query, CROPS, scorer=fuzz.partial_ratio)
        if match and match[1] >= 80:
            crop = match[0]

    if not disease:
        match = process.extractOne(query, DISEASES, scorer=fuzz.partial_ratio)
        if match and match[1] >= 80:
            disease = match[0]

    if not disease: 
        disease = "Healthy"
        
    return crop, disease
