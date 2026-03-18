from typing import Dict, Any
import time

class RAGPipeline:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.history = []  # Store query history

    def query_simple(self, question: str,top_k=3) -> Any:
        ## retriever the context
        results=self.retriever.retrieve(question,top_k=top_k)
        context="\n\n".join([doc['content'] for doc in results]) if results else ""
        if not context:
            return "No relevant context found to answer the question."

        ## generate the answwer using GROQ LLM
        prompt=f"""Use the following context to answer the question concisely.
            Context: {context}
            Question: {question}
            Answer:"""

        response=self.llm.invoke([prompt.format(context=context,query=question)])
        return response.content

    # --- Enhanced RAG Pipeline Features ---
    def query_advanced(self, query, top_k=5, min_score=0.2, return_context=False):
        """
        RAG pipeline with extra features:
        - Returns answer, sources, confidence score, and optionally full context.
        """
        results = self.retriever.retrieve(query, top_k=top_k, score_threshold=min_score)
        if not results:
            return {'answer': 'No relevant context found.', 'sources': [], 'confidence': 0.0, 'context': ''}
        
        # Prepare context and sources
        context = "\n\n".join([doc['content'] for doc in results])
        sources = [{
            'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),
            'page': doc['metadata'].get('page', 'unknown'),
            'score': doc['similarity_score'],
            'preview': doc['content'][:300] + '...'
        } for doc in results]
        confidence = max([doc['similarity_score'] for doc in results])
        
        # Generate answer
        prompt = f"""Use the following context to answer the question concisely.\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"""
        response = self.llm.invoke([prompt.format(context=context, query=query)])
        
        output = {
            'answer': response.content,
            'sources': sources,
            'confidence': confidence
        }
        if return_context:
            output['context'] = context
        return output

    def query_formatted(self, question: str, top_k: int = 5, min_score: float = 0.2, stream: bool = False, summarize: bool = False) -> Dict[str, Any]:
        # Retrieve relevant documents
        results = self.retriever.retrieve(question, top_k=top_k, score_threshold=min_score)
        if not results:
            answer = "No relevant context found."
            sources = []
            context = ""
        else:
            context = "\n\n".join([doc['content'] for doc in results])
            sources = [{
                'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),
                'page': doc['metadata'].get('page', 'unknown'),
                'score': doc['similarity_score'],
                'preview': doc['content'][:120] + '...'
            } for doc in results]
            # Streaming answer simulation
            prompt = f"""Use the following context to answer the question concisely.\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"""
            if stream:
                print("Streaming answer:")
                for i in range(0, len(prompt), 80):
                    print(prompt[i:i+80], end='', flush=True)
                    time.sleep(0.05)
                print()
            response = self.llm.invoke([prompt.format(context=context, question=question)])
            answer = response.content

        # Add citations to answer
        citations = [f"[{i+1}] {src['source']} (page {src['page']})" for i, src in enumerate(sources)]
        answer_with_citations = answer + "\n\nCitations:\n" + "\n".join(citations) if citations else answer

        # Optionally summarize answer
        summary = None
        print("<<<<<<<<<<<<---------------" + str(summarize))
        print("<<<<<<<<<<<<---------------" + str(answer))
        if summarize and answer:
            summary_prompt = f"Summarize the following answer in 2 sentences:\n{answer}"
            summary_resp = self.llm.invoke([summary_prompt])
            summary = summary_resp.content

        # Store query history
        self.history.append({
            'question': question,
            'answer': answer,
            'sources': sources,
            'summary': summary
        })

        return {
            'question': question,
            'answer': answer_with_citations,
            'sources': sources,
            'summary': summary,
            'history': self.history
        }