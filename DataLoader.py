import pandas as pd  
import os  
import json  
import jq  
from langchain.docstore.document import Document  
import warnings  
warnings.filterwarnings("ignore")  
  
def GetCSV(outputPath):  
    if not os.path.exists(outputPath):  
        df = pd.DataFrame(columns=["id", "groudTruthContext", "retrievedContext", "questionId" ,"question", "answer"])  
        return df  
    else:  
        df = pd.read_csv(outputPath)  
        return df  
  
def LoadQuestions(questionsPath):  
    questions_dict = {}  
    with open(questionsPath, 'r', encoding='utf-8') as file:  
        for line in file:  
            question = json.loads(line.strip())  
            questions_dict[question['_id']] = question['text']  
    return questions_dict  
  
def LoadMappings(mappingsPath):  
    mappings_df = pd.read_csv(mappingsPath, sep='\t')  
    mappings_df['corpus-id'] = mappings_df['corpus-id'].astype(str)  
    return mappings_df
  
def LoadContext(sourcePath, outputPath, questionsPath, mappingsPath, numLines=None, save_df=True):  
    print("Loading data...")
    df = GetCSV(outputPath)  
    questions_dict = LoadQuestions(questionsPath)  
    mappings_df = LoadMappings(mappingsPath)  
    documents = []  
    jq_schema = jq.compile("{_id: ._id, text: .text}")  
    valid_lines = 0  
    
    print("Processing files...")
    with open(sourcePath, 'r', encoding='utf-8') as file:  
        for line in file:  
            if numLines is not None and valid_lines >= numLines:  
                break  

            document = json.loads(line.strip())  
            parsed_document = jq_schema.input(document).first()  

            corpus_id = str(parsed_document['_id'])
            query_id_row = mappings_df.loc[mappings_df['corpus-id'] == corpus_id, 'query-id']

            if not query_id_row.empty and save_df:  
                query_id = query_id_row.values[0]  
                question = questions_dict.get(str(query_id), None)  
  
                if question:  
                    new_row = pd.DataFrame([{"id": parsed_document['_id'], "groudTruthContext": parsed_document['text'], "questionId": query_id_row, "question": question}])  
                    df = pd.concat([df, new_row], ignore_index=True)  
                    valid_lines += 1  
            
            doc = Document(page_content=parsed_document['text'], metadata={"_id": parsed_document['_id']})  
            documents.append(doc)  
      
    if save_df:
        df.to_csv(outputPath, index=False)
    print("Data processing complete.")  
    return documents  
  
  
if __name__ == "__main__":  
    sourcePath = "Data/fiqa/corpus.jsonl"  
    questionsPath = "Data/fiqa/queries.jsonl"  
    mappingsPath = "Data/fiqa/qrels/train.tsv"  
    numLines = 100
 
    documents = LoadContext(sourcePath, "Data/PreparedData/EmbeddingRetriever.csv", questionsPath, mappingsPath, numLines)  
    documents = LoadContext(sourcePath, "Data/PreparedData/HeuristicRetriever.csv", questionsPath, mappingsPath, numLines) 
    documents = LoadContext(sourcePath, "Data/PreparedData/FrequentistRetriever.csv", questionsPath, mappingsPath, numLines) 