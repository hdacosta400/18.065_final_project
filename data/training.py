import numpy as np
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras import callbacks
from tensorflow import keras
import os
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
import cv2

'''
Takes skeleton, computes features on it and feeds those features into model
'''
class PostureFeatureModel:
    def __init__(self, data):
        self.data = data
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_validation = None
        self.y_validation = None 

    '''
    load the data to be used in training
    '''
    def load_data(self):
        dataset = loadtxt(self.data, delimiter=',')

        number_of_features = 5
        inps = dataset[:,0:number_of_features] # nx5 , posture features
        labels = dataset[:,number_of_features] # posture labels
        # print("inps:", inps)
        # print("labels:", labels)
        print("test:", self.x_test)

        print(np.shape(inps[:1]), inps[:1])
        return inps, labels
    
    '''
    apply transformation to data given the x and y values
    '''
    def transform_data(self):
        inps, labels = self.load_data()
        trisect = len(inps)//3
        # split data
        self.x_train = inps[:2*trisect]
        self.y_train = labels[:2*trisect]

        self.x_test = inps[2*trisect:3*trisect-10]
        self.y_test = labels[2*trisect:3*trisect-10]

        self.x_validation = inps[3*trisect-10:]
        self.y_validation = labels[3*trisect-10:]
        
        
    '''
    Train the model
    '''
    def train_model(self):
        self.transform_data()
        # Define Sequential model with 3 layers
        model = Sequential()
        model.add(Dense(12, input_dim=5, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        print("compiling model.....")
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[keras.metrics.BinaryAccuracy( name="binary_accuracy", dtype=None, threshold=0.5)])

        #train the model
        # include callback to prevent overfitting
        earlystop = callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True, verbose=1)

        model.fit(self.x_train, self.y_train, epochs=8, batch_size=10, verbose=1,
                  validation_data=(self.x_validation, self.y_validation), callbacks=[earlystop])
        model.save('posture_model')

    '''
    Test the model
    '''
    def evaluate_model(self):
        model = keras.models.load_model('posture_model')
        _, accuracy = model.evaluate(self.x_test, self.y_test)
        print('Test Accuracy: %.2f' % (accuracy*100))

        # _, accuracy = model.evaluate(self.x_validation, self.y_validation)
        # print('Validation Accuracy: %.2f' % (accuracy*100))
        return accuracy

    def prediction_error(self):
        '''
            Predict posture from new images, get mean squared error
        '''
        model = keras.models.load_model('posture_model')
        prediction = model.predict(self.x_validation)
        print("prediction:", prediction)
        mse = np.square(np.subtract(prediction,self.y_validation)).mean()
        print("MSE:", mse)
        return mse



'''
Feeds skeletons into model

images are RGB 1080 x 720

following https://towardsdatascience.com/input-pipeline-for-images-using-keras-and-tensorflow-c5e107b6d7b9


'''

    
if __name__ == "__main__":
    # check current directory and make sure you're in the right one
    # current_directory = os.getcwd()
    # file_directory = os.path.dirname(__file__)

    # if current_directory != file_directory:
    #     os.chdir(file_directory)


    p = PostureFeatureModel('./training_data.csv')
    p.train_model()
    p.evaluate_model()
    # p.prediction_error()

    pass