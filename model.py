import os
import tensorflow as tf
import numpy as np

class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, 'bestia'))
        self.model2 = tf.keras.models.load_model(os.path.join(path, 'lazza'))
        self.model3 = tf.keras.models.load_model(os.path.join(path, 'stirato'))

    def predict(self, X):

        for x in X:
            x = (x/255)
        
        # Note: this is just an example.
        # Here the model.predict is called, followed by the argmax
        out1 = self.model.predict(X)
        out2 = self.model2.predict(X)
        out3 = self.model3.predict(X)
        
        out1 = tf.argmax(out1, axis=-1)  # Shape [BS]
        out2 = tf.argmax(out2, axis=-1)  # Shape [BS]
        out3 = tf.argmax(out3, axis=-1)  # Shape [BS]

        real_out = tf.zeros(len(out1))          #creo un tensore di zeri
        real_out = tf.Variable(real_out)        #rendo modificabile il tensore
        sum = tf.add(tf.add(out1,out2),out3)    #creo un tensore somma dei tre di output
        for i in range(len(out1)):
            if(sum[i] > 1):
                real_out[i].assign(1)           #modifico il tensore 
        real_out = real_out.read_value()        #ritrasformo il tensore di output da variable a tensor

        return real_out
    

