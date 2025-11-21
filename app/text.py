import os
from dotenv import load_dotenv
import numpy as np
from langchain_neo4j import Neo4jGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from PIL import Image
from utils import ANSWER_GEN_TEMPLATE, CYPHER_SUBGRAPH, cleanup_relationships
from vision import predict_fn, id_to_class, generate_grad_rollout
import numpy as np
load_dotenv()

graph = Neo4jGraph(url="neo4j://127.0.0.1:7687", username=os.getenv("NEO4J_USER"), password=os.getenv("NEO4J_PASSWORD"), enhanced_schema=True)

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

ANSWER_GEN_PROMPT = PromptTemplate.from_template(ANSWER_GEN_TEMPLATE)
answer_chain = ANSWER_GEN_PROMPT | llm


def run_pipeline(image, query):

    try:
        image_pil = Image.open(image).convert("RGB")
        image_np = np.array(image_pil)

        probs = predict_fn([image_np])
        predicted_class_id = np.argmax(probs[0])
        predicted_class_name = id_to_class[predicted_class_id]
    
        crop, dis = predicted_class_name.split(',')
        subgraph = graph.query(CYPHER_SUBGRAPH, params={"disease": dis, "crop" : crop })[0]['rels']
        subgraph = cleanup_relationships(subgraph)
        answer = answer_chain.invoke({
            "question": query,
            "context": str(subgraph),
            "crop": crop,
            "disease": dis
        })
        return answer.content, subgraph
        
    except FileNotFoundError:
        print(f"Error: Image file not found at '{image}'")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure your image file is a valid format (JPG, PNG, etc.)")






