class GenerateAnswer():
    def __init__(self, time, generator):
        self.time = time
        self.generator = generator

    def generate_answer(self, prompt_template, question, context):
        start_time = self.time.time()
        prompt = prompt_template.format(context=context, question=question)
        checkpoint_1 = self.time.time() - start_time
        print(f"Checkpoint 1: {checkpoint_1:.4f} seconds")

        start_time = self.time.time()
        generator = self.generator(model="HuggingFaceTB/SmolLM-1.7B-Instruct",
                                        task="text-generation",
                                        generation_kwargs={
                                            "max_new_tokens": 150,
                                            "do_sample": False,
                                        })
        checkpoint_2 = self.time.time() - start_time
        print(f"Checkpoint 2: {checkpoint_2:.4f} seconds")

        start_time = self.time.time()
        generator.warm_up()
        checkpoint_3 = self.time.time() - start_time
        print(f"Checkpoint 3: {checkpoint_3:.4f} seconds")

        start_time = self.time.time()
        response = generator.run(prompt)
        checkpoint_4 = self.time.time() - start_time
        print(f"Checkpoint 4: {checkpoint_4:.4f} seconds")

        replies = response['replies']

        generate_time = sum([checkpoint_1, checkpoint_2, checkpoint_3, checkpoint_4])
        print(f"Total answer generation time: {generate_time:.4f} seconds")
        print(replies)
        return replies