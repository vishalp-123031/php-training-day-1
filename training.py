from django.core.management.base import BaseCommand
from langchain_community.document_loaders import DedocAPIFileLoader
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_qdrant import FastEmbedSparse, RetrievalMode
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_community.document_transformers import LongContextReorder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os

class Command(BaseCommand):
  help = "RAG Based BOT."

  def handle(self, *args, **options):
    loader = DedocAPIFileLoader(
      "/app/uploads/documents/1010_employee_policy_retriever_tool/D182_Policy No 09_Employee Leave Policy.pdf",
      url=os.environ.get('DEDOC_API_ENDPOINT'),
      need_header_footer_analysis=True,
      need_pdf_table_analysis=True,
      pages="4:",
      pdf_with_text_layer="tabby"
    )
    elements = loader.load_and_split()

    #for element in elements:
    #  self.stdout.write(self.style.SUCCESS(element.page_content))
    #  self.stdout.write(self.style.SUCCESS("==============================="))

    embedding_model = OllamaEmbeddings(
      model=os.environ.get('EMBEDDING_MODEL_NAME'),
      base_url=os.environ.get('OLLAMA_BASE_URL'),
    )
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    vectorstore = QdrantVectorStore.from_documents(
      elements,
      embedding=embedding_model,
      sparse_embedding=sparse_embeddings,
      url=os.environ.get("QDRANT_BASE_URL"),
      prefer_grpc=True,
      collection_name="training_20250212",
      retrieval_mode=RetrievalMode.HYBRID,
    )

    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(
      model_name=os.environ.get('LLM_NAME'),
      openai_api_base=os.environ.get('VLLM_BASE_URL'),
      openai_api_key="EMPTY",
      temperature=0,
      max_tokens=os.environ.get('MAX_TOKEN_LEN')
    )

    retriever_from_llm = MultiQueryRetriever.from_llm(
      retriever=retriever,
      llm=llm
    )

    query = "What is privilege leave?"
    # result = retriever_from_llm.invoke(query)

    compressor = FlashrankRerank()
    compression_retriever = ContextualCompressionRetriever(
      base_compressor=compressor,
      base_retriever=retriever_from_llm
    )
    docs = compression_retriever.invoke(query)
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(docs)

    prompt = ChatPromptTemplate.from_messages(
      [
        ("system", """
         You are expert AI finding result in the provided context.
         Your task is to find the answer in the provided cotext.
         Provide accurate & concise answer in maximum three sentences.
         Given below is the context.
         {context}
        """
        ),
        ("human", "{question}"),
      ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    result = question_answer_chain.invoke({"question": query, "context": reordered_docs})

    print(result)
    # for element in reordered_docs:
    #   self.stdout.write(self.style.SUCCESS(element.page_content))
    #   self.stdout.write(self.style.SUCCESS("==============================="))





