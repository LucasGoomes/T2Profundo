import os
import pandas as pd
import time
from langchain_ollama.llms import OllamaLLM

prompt_template = "Respond to the user's question using only the context provided. Return only the response, nothing else. If the context does not contain the answer, respond that you don't know the answer. Question: {question} Context: {context}"
model = OllamaLLM(model="deepseek-r1:1.5b")

def GenerateResponses(df_path, retrieverName):
    df = pd.read_csv(df_path)
    for index, row in df.iterrows():  
        print(f"Processing question id {row['questionId']}")
        question = row['question']
        print(f"Question: {question}")
        context= row['retrievedContext']
        prompt = prompt_template.format(question=question, context=context)
        response = model.invoke(prompt)
        response = remove_think_section(response)
        print(f"Response: {response}")
        df.at[index, 'answer'] = response
    df.to_csv(f"Data/PreparedData/{retrieverName}_WithAnswers.csv", index=False)

def remove_think_section(text):  
    start_think = text.find("<think>")  
    end_think = text.find("</think>") + len("</think>")  
    
    if start_think != -1 and end_think != -1:  
        text = text[:start_think] + text[end_think:]  
    return text.strip()  

if __name__ == "__main__":
    #GenerateResponses("Data/PreparedData/EmbeddingRetriever.csv", "EmbeddingRetriever")
    #GenerateResponses("Data/PreparedData/HeuristicRetriever.csv", "HeuristicRetriever")
    GenerateResponses("Data/PreparedData/FrequentistRetriever.csv", "FrequentistRetriever")
