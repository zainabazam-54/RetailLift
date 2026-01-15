import cv2
import numpy as np
import tensorflow as tf

class ShopliftingClassifier:
    def __init__(self,model_path,threshold=0.5): #constructor
        self.model=tf.keras.models.load_model("src/models/efficientNetModel.h5")
        self.threshold=threshold  #ignore threshold below this value

    def preprocess_frame(self,img):
        img=cv2.resize(img,(224,224))
        img=img/255.0
        img=np.expand_dims(img,axis=0)
        return img
    
    def predict(self,person_crop):
        if person_crop.size==0:  #return normal if no person detected
            return "Normal",0.0
        
        inp = self.preprocess_frame(person_crop)
        prob=self.model.predict(inp,verbose=0)[0][0]
        label="Shoplifting" if prob>self.threshold else "Normal"
        return label,float(prob)
