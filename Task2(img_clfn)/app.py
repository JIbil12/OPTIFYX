import tkinter as tk
from tkinter import *
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('mnist_cnn_model.h5')

def preprocess_image(img):
    # Resize the image to 28x28
    img = img.resize((28, 28))
    
    # Convert to grayscale
    img = img.convert('L')
    
    # Invert the image (black background to white background)
    img = ImageOps.invert(img)
    
    # Convert to numpy array and normalize
    img_array = np.array(img).astype('float32') / 255.0
    
    # Reshape to match the model's expected input shape
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array

def predict_digit(img):
    img_array = preprocess_image(img)
    
    # Predict the digit
    prediction = model.predict(img_array)
    
    return np.argmax(prediction)

class PaintApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Draw a Digit')
        
        self.canvas = Canvas(self.root, width=280, height=280, bg='white')
        self.canvas.pack()
        
        self.button_predict = Button(self.root, text="Predict", command=self.predict)
        self.button_predict.pack()
        
        self.button_clear = Button(self.root, text="Clear", command=self.clear_canvas)
        self.button_clear.pack()
        
        self.label_result = Label(self.root, text="Draw a digit and click Predict")
        self.label_result.pack()
        
        self.canvas.bind("<B1-Motion>", self.paint)
        
        self.image = Image.new("L", (280, 280), color=255)
        self.draw = ImageDraw.Draw(self.image)
    
    def paint(self, event):
        x1, y1 = (event.x - 7), (event.y - 7)
        x2, y2 = (event.x + 7), (event.y + 7)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=14)
        self.draw.ellipse([x1, y1, x2, y2], fill=0, width=14)
    
    def predict(self):
        # Get the predicted digit
        digit = predict_digit(self.image)
        self.label_result.config(text=f"Predicted Digit: {digit}")
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), color=255)
        self.draw = ImageDraw.Draw(self.image)

if __name__ == '__main__':
    root = Tk()
    app = PaintApp(root)
    root.mainloop()