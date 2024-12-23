class GetRelevantContext:
    def __init__(self, time, cosine_similarity):
        self.time = time
        self.cosine_similarity = cosine_similarity

    def get_relevant_context(self, question, embeddings, embed_text, dataset, top_k=5):
        context_start = self.time.time()
        question_embedding = embed_text(question)
        similarities = self.cosine_similarity([question_embedding], embeddings)
        top_indices = similarities.argsort()[0][-top_k:][::-1]
        top_context = [dataset[int(i)]['content'] for i in top_indices]
        context_time = self.time.time() - context_start
        print(f"Context retrieval time: {context_time:.4f} seconds")
        print('\n'.join(repr(item) for item in top_context))
        return top_context