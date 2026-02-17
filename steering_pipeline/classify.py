from sklearn import svm
import pickle

filename = 'your_svm_model.pkl'

with open(filename, 'rb') as f:
    loaded_model = pickle.load(f)

def classify(out_file, steering_vector=None):
    

