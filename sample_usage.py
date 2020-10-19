"""
sample_usage.py
Created on Oct 18 2020 15:05
@author: Moayed Haji Ali mali18@ku.edu.tr

"""
import numpy as np
from basic_usage.sketchformer import continuous_embeddings
import time
import warnings
import random

warnings.filterwarnings("ignore")


class Basic_Test:

    def __init__(self):
        # prepare sample data from the quickdraw
        filename = "basic_usage/quickdraw_samples/sketchrnn_apple.npz"
        self.apples = np.load(filename, encoding='latin1', allow_pickle=True)
        filename = "basic_usage/quickdraw_samples/sketchrnn_baseball.npz"
        self.baseball = np.load(filename, encoding='latin1', allow_pickle=True)

    def performe_test(self, model):
        print("Performing tests:")
        # extract sample embedding of N apples, and M baseballs and observe the distances
        N_apple = N_baseball = 2

        apple_sketch, baseball_sketch = [], []
        apple_embedding, baseball_embedding = [], []

        # get random N_apple sketches 
        for _ in range(N_apple):
            ind = random.randint(0, len(self.apples['test']) - 1)
            apple_sketch.append(self.apples['test'][ind])

        # get random N_baseball sketches 
        for _ in range(N_baseball):
            ind = random.randint(0, len(self.baseball['test']) - 1)
            baseball_sketch.append(self.baseball['test'][ind])

        embeddings = model.get_embeddings(np.concatenate((apple_sketch, baseball_sketch)))

        apple_embedding = embeddings[:N_apple]
        baseball_embedding = embeddings[N_apple:]
        for i, apple_emb1 in enumerate(apple_embedding):
            for j, apple_emb2 in enumerate(apple_embedding):
                if i > j:
                    print("[Apple {} - Apple {}] embedding vectors norm: ".format(i, j), np.linalg.norm(apple_emb1 - apple_emb2))

        for i, base_emb1 in enumerate(baseball_embedding):
            for j, base_emb2 in enumerate(baseball_embedding):
                if i > j:
                    print("[Baseball {} - Baseball {}] embedding vectors norm: ".format(i, j), np.linalg.norm(base_emb1 - base_emb2))

        for i, apple_emb in enumerate(apple_embedding):
            for j, base_emb in enumerate(baseball_embedding):
                    print("[Apple {} - Baseball {}] embedding vectors norm: ".format(i, j), np.linalg.norm(apple_emb - base_emb))

        # classify sample apple sketch
        pred_class = model.classify(np.concatenate((apple_sketch, baseball_sketch)))

        for i in range(N_apple):
            print("Predicted class for a Apple sketch: ", pred_class[i])
        
        for i in range(N_baseball):
            print("Predicted class for a Baseball sketch: ", pred_class[N_apple + i])
        # classify sample baseball sketch

    def pre_trained_model_test(self):
        """peforme tests on the pretrained model
        """
        # obtain the pre-trained model
        sketchformer = continuous_embeddings.get_pretrained_model()
        self.performe_test(sketchformer)
    
    def new_model_test(self):
        """peforme tests on a new model trained on two classes (Apple, Baseball)
        """
        # train a new model
        print("Training a new model")
        MODEL_ID = "my_new_model"
        OUT_DIR = "basic_usage/pre_trained_model"
        sketches_x = np.concatenate((self.apples['train'], self.baseball['train']))
        sketches_y = np.concatenate((np.zeros(len(self.apples['train'])), np.ones(len(self.baseball['train']))))
        new_model = continuous_embeddings(sketches_x, sketches_y, ['apple', 'baseball'], MODEL_ID, OUT_DIR, resume=False)
        self.performe_test(new_model)

        # using the embedding from the checkpoint
        print("Obtain the embeddings from the stored checkpoint of the new model")

        resume_model = continuous_embeddings([], [], ['apple', 'baseball'], MODEL_ID, OUT_DIR, resume=True)
        self.performe_test(resume_model)


Basic_Test().pre_trained_model_test()

Basic_Test().new_model_test()