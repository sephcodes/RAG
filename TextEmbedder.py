class TextEmbedder:
    def __init__(self, torch, tqdm, tokenizer, base_model):
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.torch = torch
        self.tqdm = tqdm

    def embed_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
        with self.torch.no_grad():
            outputs = self.base_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    
    def embed_dataset(self, dataset):
        embeddings = []
        for item in self.tqdm(dataset, desc="Embedding documents"):
            embedding = self.embed_text(item['content'])
            embeddings.append(embedding)

        print(f"Embedded {len(embeddings)} documents.")
        print(f"Embedding shape: {embeddings[0].shape}")
        return embeddings