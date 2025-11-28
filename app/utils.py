def cleanup_relationships(raw):
    unique_simplified_tuples = set()
    for d1, r, d2 in raw:
        simplified_tuple = (d1['name'], r, d2['name']) 
        unique_simplified_tuples.add(simplified_tuple)

    deduplicated_list = []
    for name1, r, name2 in unique_simplified_tuples:
        original_format_tuple = ({'name': name1}, r, {'name': name2})
        deduplicated_list.append(original_format_tuple)

    return deduplicated_list


CYPHER_SUBGRAPH = """
MATCH (d:Entity {name: $disease})
MATCH (c:Entity {name: $crop})
MATCH (d)-[rA:AFFECTS]->(c)

OPTIONAL MATCH (d)-[rPRESENTS:PRESENTS]->(s:Entity)
OPTIONAL MATCH (d)-[rPATHOGEN:CAUSED_BY]->(p:Entity)
OPTIONAL MATCH (d)-[rTREAT:TREATED_BY]->(t:Entity)

WITH 
     COLLECT(DISTINCT d) +
     COLLECT(DISTINCT c) +
     COLLECT(DISTINCT s) +
     COLLECT(DISTINCT p) +
     COLLECT(DISTINCT t) AS nodes,
     COLLECT(rA) + 
     COLLECT(rPRESENTS) +
     COLLECT(rPATHOGEN) + 
     COLLECT(rTREAT) AS rels

RETURN nodes, rels
"""

ANSWER_GEN_TEMPLATE = """
### Role
You are an expert AI agricultural assistant. Your job is to provide a **precise, scientifically reliable, and user-focused** answer to the user's question about their plant.

### Information Provided
You are given the following:

1. **[Original User Question]** — This is what you must answer.
2. **[Diagnosis]** — The disease or condition detected from the image (e.g., a specific disease or "Healthy").
3. **[Retrieved Knowledge]** — A small, relevant knowledge subgraph about the diagnosis (crop → disease → symptoms → treatments).

### Consistency Check (Important)
Before answering:
- Compare the **crop and disease** extracted from the question with the **Diagnosis** and **Provided Context**.
- If they **do not match**, DO NOT provide treatment or plant guidance.
- Instead, respond with a short message telling the user that the crop or symptom they asked about does not match the diagnosis you have, and ask them to recheck or upload the correct image.

### How to Use the Information
If the question and diagnosis **match**:
- Use **[Diagnosis]** to determine *what is happening* with the plant.
- Use the **context** to provide:
  - brief scientific justification,
  - symptom reasoning,
  - actionable recommendations,
  - or reassurance if the plant is healthy.
- You are NOT allowed to question, re-verify, or contradict the diagnosis.

### Critical Rules (Follow Strictly)
1. **Directly answer the user's question** — this is your top priority.
2. Your answer must be:
   - **specific and to the point**,
   - **scientifically grounded**,
   - **clear and concise**,
   - **actionable when needed**.
3. For "what/why/how" questions:
   - Provide a short explanation using the provided information.
4. **Do NOT**:
   - Mention the knowledge graph itself.
   - Say "I cannot confirm" or be uncertain.
   - Add unrelated information.
   - Give unsafe chemical advice.

### Output Style
- If mismatch: 1-2 sentences informing the user of the mismatch.
- If matched: 2-4 sentences maximum, expert tone.
- Avoid lists unless absolutely necessary.

Crop: {crop}
Diagnosis: {disease}

<context>
{context}
</context>

Question: {question}

Answer:
"""

CYPHER_SUBGRAPH_H = """
        MATCH (d:Entity)-[r:AFFECTS]->(c:Entity {name: $crop})
        RETURN DISTINCT 
            d.name AS disease,
            type(r) AS relationship,
            c.name AS crop
        """
CYPHER_SUBGRAPH_D =  """
    MATCH (d:Entity {name: $disease})
    OPTIONAL MATCH (d)-[rA:AFFECTS]->(c:Entity)
    OPTIONAL MATCH (d)-[rPRESENTS:PRESENTS]->(s:Entity)
    OPTIONAL MATCH (d)-[rPATHOGEN:CAUSED_BY]->(p:Entity)
    OPTIONAL MATCH (d)-[rTREAT:TREATED_BY]->(t:Entity)

    WITH 
        COLLECT(DISTINCT d) +
        COLLECT(DISTINCT c) +
        COLLECT(DISTINCT s) +
        COLLECT(DISTINCT p) +
        COLLECT(DISTINCT t) AS nodes,
        COLLECT(rA) + 
        COLLECT(rPRESENTS) +
        COLLECT(rPATHOGEN) + 
        COLLECT(rTREAT) AS rels,
        d

    RETURN nodes, rels
"""


