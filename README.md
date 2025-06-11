install depencies
generate evaluantions csv running DataLoader.py - it will populate id,groundthruthContext text and question
run RetrieveContext to retrieve the context based on the question and populate retrievedContext on the csv
run LLM.py to generate the responses based on the context retrieved by RetrieveContext

### Slide: Estratégias Implementadas  
  
#### Estratégias Heurísticas  
- **Heurística de Busca por Palavras**:  
  - Leitura do arquivo CSV com perguntas e contextos verdadeiros.  
  - Iteração sobre as perguntas e contextos para determinar relevância.  
  - Contagem de palavras comuns: palavras presentes tanto na pergunta quanto no contexto verdadeiro.  
  - Seleção do contexto com maior contagem de palavras correspondentes.  
  
#### Estratégias com Embeddings  
- **Utilização de Embeddings**:  
  - **SentenceTransformerEmbeddings**:  
    - Uso do modelo "all-MiniLM-L6-v2" para transformar textos em vetores de embeddings.  
    - Verificação de existência de embeddings previamente gerados e armazenados.  
    - Geração de novos embeddings a partir dos documentos, se necessário.  
    - Persistência dos embeddings em diretório específico para reutilização.  
    - **Similaridade Usada**: `retriever.similarity_search(question, k=1)`:  
      - Busca de similaridade utilizando a métrica de similaridade de cosseno.  
      - Seleção do contexto com maior similaridade de cosseno em relação à pergunta.  
  
#### Estratégias com BoW e Outras Abordagens Frequentistas  
- **TF-IDF (Term Frequency-Inverse Document Frequency)**:  
  - **TFIDFRetriever**:  
    - Cálculo da importância das palavras com base na frequência e raridade nos documentos.  
    - Transformação dos textos em vetores TF-IDF.  
    - **Similaridade Usada**: Similaridade de cosseno entre os vetores TF-IDF.  
    - Seleção do contexto mais relevante com base na similaridade entre vetores TF-IDF das perguntas e dos contextos verdadeiros.  