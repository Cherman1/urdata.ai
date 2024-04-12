from flask import Flask, request, jsonify
from pinecone_text.sparse import SpladeEncoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone
import time
from langchain_openai import OpenAIEmbeddings

app = Flask(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINE_API_KEY = os.getenv("PINE_API_KEY")
index_name = "splade"
embed = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=OPENAI_API_KEY, dimensions=768)
pc = Pinecone(api_key=PINE_API_KEY)

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query')
    namespace_name = data.get('namespace', 'default_namespace')

    index = pc.Index(index_name)
    time.sleep(0.5)  # Ensure the index connection is established

    splade_encoder = SpladeEncoder()
    retriever = PineconeHybridSearchRetriever(
        embeddings=embed,
        sparse_encoder=splade_encoder,
        index=index,
        namespace=namespace_name
    )

    results = retriever.get_relevant_documents(query)
    # Convert each Document object to a dictionary
    result_dicts = [result.__dict__ for result in results]

    return jsonify(result_dicts)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
