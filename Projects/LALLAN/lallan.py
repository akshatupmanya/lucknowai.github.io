import numpy as np
import os
from sentence_transformers import SentenceTransformer
from lucknowllm import UnstructuredDataLoader, split_into_segments
import google.generativeai as gen_ai
ROLE = """
You are an expert informator system about Lucknow, I'll give you question and context and you'll return the answer in a sweet and sarcastic tone and in a respectful manner. You will use Hum instead of main. Your name is Lallan. The full form of Lallan is "Lucknow Artificial Language and Assistance Network". Call only Janab-e-Alaa instead of phrase My dear Friend. Say Salaam Miya! instead of Greetings.
"""

class DataLoader:
    def _init_(self):
        self.loader = UnstructuredDataLoader()
        self.base_path = self.loader.base_path
        self.subfolder_names = [
            "Art_and_Literature", 
            "Arts_and_Crafts", 
            # Add other subfolder names
        ]
        self.all_data = {}

    def get_data_from_subfolder(self, subfolder_name):
        data = self.loader.get_data(folder_name=subfolder_name)
        self.all_data[subfolder_name] = data
        for subfolder in os.listdir(os.path.join(self.base_path, subfolder_name)):
            if os.path.isdir(os.path.join(self.base_path, subfolder_name, subfolder)):
                self.get_data_from_subfolder(os.path.join(subfolder_name, subfolder))

    def load_data(self):
        for subfolder_name in self.subfolder_names:
            self.get_data_from_subfolder(subfolder_name)

class ModelInitializer:
    def _init_(self):
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.loader = DataLoader()
        self.loader.load_data()
        self.external_database = self._get_external_database()

    def _get_external_database(self):
        external_database = []
        for subfolder_name, data in self.loader.all_data.items():
            for data_item in data:
                external_database.append(f"File Name: {data_item['file_name']}")
                external_database.append(f"Data: {data_item['data']}")
        return ' '.join([str(elem) for elem in external_database])

    def preprocess_data(self):
        chunks = split_into_segments(self.external_database)
        embedded_data = self.model.encode(chunks)
        return chunks, embedded_data

class GenerativeModel:
    def _init_(self):
        self.generation_config = {
            "temperature": 0.1,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 2084,
        }
        self.gemini = gen_ai.GenerativeModel(model_name="gemini-1.0-pro",
                                              generation_config=self.generation_config)

    def generate_response(self, argumented_prompt):
        model_output = self.gemini.generate_content(contents=argumented_prompt)
        return model_output.text

class Lallan:
    def _init_(self):
        self.model_initializer = ModelInitializer()
        self.chunks, self.embedded_data = self.model_initializer.preprocess_data()
        self.generative_model = GenerativeModel()
        self.prompt_1, self.response_1 = "", ""

    def run_lallan(self, prompt_2):
        queries = [prompt_2]
        embedded_queries = self.model_initializer.model.encode(queries)
        for i, query_vec in enumerate(embedded_queries):
            similarities = cosine_similarity(query_vec[np.newaxis, :], self.embedded_data)
            top_indices = np.argsort(similarities[0])[::-1][:3]
            top_doct = [self.chunks[index] for index in top_indices]
            argumented_prompt = f'''
            Role: {ROLE}
            Query: {self.prompt_1}
            response: {self.response_1}
            Query : {queries[i]}
            Contexts : {top_doct[0]}
            response:
            '''
            model_output = self.generative_model.generate_response(argumented_prompt)
            self.response_2 = print("\n", "LALLAN:" + model_output.text)
            self.prompt_1, self.response_1 = prompt_2, self.response_2

if _name_ == "_main_":
    try:
        print("Namaskar Adaab Parnaam! Hum Lallan from Lucknow, Aur Humse Lucknow ke baare me Kuch bhi Poochiye\n\nYOU:")
        lallan = Lallan()
        while True:
            lallan.run_lallan(input())
    except:
        print("Maharaj lagta hai kuch ant shant daal diye hain aap! Iske baare me to kuch nahi bata sakte...")
