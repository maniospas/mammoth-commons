from kfp import dsl
import numpy as np
import onnxruntime as rt
from io import BytesIO
import urllib.request
import zipfile

class Model:
    integration = dsl.Model
    
class ONNXEnsemble:
    def __init__(self, zip_url):
        self.zip_url = zip_url
        self.models, self.params = self._load_models_and_params_from_zip()

    def _extract_number(self, filename):
        match = re.search(r'_(\d+)\.onnx$', filename)
        return int(match.group(1)) if match else float('inf')
        
    def _load_models_and_params_from_zip(self):
        # Download the zip file into memory
        #with urllib.request.urlopen(self.zip_url) as response:
        #    zip_file_bytes = response.read()

        models = []
        model_names=[]
        params = None
        
        def myk(name):
            return int(re.findall(r'[+-]?\d+',name)[0])

        # Read the zip file
        with zipfile.ZipFile(self.zip_url) as myzip:
            # Extract and load the weights file
            for file_name in myzip.namelist():
                if file_name.endswith('.npy'):
                    with myzip.open(file_name) as param_file:
                        params= np.load(param_file,allow_pickle=True)
                elif file_name.endswith('.onnx'):
                    model_names.append(file_name)
            
            model_names.sort(key=myk)   
          
            for file_name in model_names:        
                with myzip.open(file_name) as model_file:
                    model_bytes = model_file.read()
                    models.append(self.ONNX(model_bytes))
        
        return models, params.item()

    class ONNX(Model):
        def __init__(self, model_bytes):
            self.sess = rt.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])

        def predict(self, x):
            input_name = self.sess.get_inputs()[0].name
            label_name = self.sess.get_outputs()[0].name
            return self.sess.run([label_name], {input_name: x.astype(np.float32)})[0]

    def predict(self, X):
        n_classes = self.params['n_classes']
        classes = self.params['classes'][:, np.newaxis]

        pred = sum((estimator.predict(X)== classes).T * w  for estimator, w in zip(self.models[:self.params['theta'] ], self.params['alphas'][:self.params['theta']]))
        pred /= self.params['alphas'][:self.params['theta']].sum()
        pred[:, 0] *= -1
        preds=classes.take(pred.sum(axis=1) > 0, axis=0)
        return np.squeeze(preds,axis=1)