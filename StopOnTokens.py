from transformers import StoppingCriteria

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids):
        # self.tokenizer = tokenizer
        # self.stop_words = ["system", "Human:", "Assistant:", "\n\n", "Question", "Answer"]
        # self.stop_ids = [self.tokenizer(stop_word, add_special_tokens=False).input_ids for stop_word in self.stop_words]
        super().__init__()
        self.stop_ids = stop_ids
    
    def __call__(self, input_ids):
        for stop_id in self.stop_ids:
            if input_ids[0][-len(stop_id):].tolist() == stop_id:
                return True
        return False