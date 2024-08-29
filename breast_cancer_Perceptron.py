import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer 

class PerceptronClassifier:
    def __init__(self):
        self.w = None
        self.w0 = None
    
    def load_dataset(self):
        data = load_breast_cancer() 
        self.x = data.data 
        self.y = data.target 
        self.y[self.y==0] = -1
    
    def distance_from_hp(self, w, w0, x):
        d = x@w + w0
        return d

    def train(self, num_iter=1000):
        self.w = np.random.normal(size=self.x.shape[1])
        self.w0 = np.random.normal()
        
        for i in range(num_iter):
            miss_classified_points = 0
            for(xi,yi) in zip(self.x,self.y):
                d = self.distance_from_hp(self.w,self.w0,xi)
                hs = np.sign(d)
                if hs != yi:
                    miss_classified_points += 1
                    self.w = self.w + xi * yi
                    self.w0 = self.w0 + yi
            accuracy = 100 - (100 * miss_classified_points / len(self.y))
            # print(f"Iteration {i+1}: Accuracy={accuracy:.2f}%")
            # print(f"Weights: w={self.w[:5]}, ..., w0={self.w0}")  
            
            if accuracy > 90:
                break
            if miss_classified_points == 0:
                return
        
    def predict(self, x_new):
        prediction = np.sign(np.dot(x_new, self.w) + self.w0)
        return prediction
    
    def classify_sample(self, x_new):
        prediction = self.predict(x_new)
        if prediction == 1:
            return "safe"
        elif prediction == -1:
            return "unsafe"
        else:
            return "unknown"



classifier = PerceptronClassifier()
classifier.load_dataset()
classifier.train()

 # Generate a random sample
new_sample = np.random.normal(size=classifier.x.shape[1]) 
print(new_sample)
classification_result = classifier.classify_sample(new_sample)
print(f"\nClassification result for new sample: {classification_result}")
