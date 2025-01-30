from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import ttk
import numpy as np
import cv2
from keras.models import load_model
from keras.optimizers import RMSprop
from ttkthemes import ThemedTk

# Create the main application window
main = ThemedTk(theme="breeze")  
main.title("Alzheimer's Disease Prediction")
main.geometry("1300x1200")

# Define class mapping
mapping = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

def load():
    global model
    try:
        model = load_model('model_final.h5')
        model.summary()
        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(learning_rate=1e-4),
                      metrics=['accuracy'])
        text.insert(END, "Model loaded successfully.\n")
    except Exception as e:
        text.insert(END, f"Error loading model: {str(e)}\n")

def upload():
    global filename
    filename = askopenfilename()
    if filename:
        text.delete('1.0', END)
        text.insert(END, "File Uploaded: " + str(filename) + "\n")
    else:
        text.insert(END, "No file selected.\n")

def imagepreprocess():
    global img4
    if filename:
        try:
            img3 = cv2.imread(filename)
            img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
            img3 = cv2.resize(img3, (224, 224))
            img4 = np.reshape(img3, [1, 224, 224, 3])
            text.insert(END, "Image preprocessed successfully.\n")
        except Exception as e:
            text.insert(END, f"Error in image preprocessing: {str(e)}\n")
    else:
        text.insert(END, "No image file to preprocess.\n")

def predict():
    if 'model' in globals() and 'img4' in globals():
        try:
            predictions = model.predict(img4)
            prediction = np.argmax(predictions[0])  # Get index of max probability
            prediction_name = mapping[prediction]
            text.insert(END, "Predicted output for uploaded Image: " + str(prediction_name) + "\n")
        except Exception as e:
            text.insert(END, f"Error in prediction: {str(e)}\n")
    else:
        text.insert(END, "Model or image data missing.\n")

# Title Label
font = ('montserrat', 20, 'bold')
title = Label(main, text="Alzheimer's Disease Prediction")
title.config(bg='black', fg='white')
title.config(font=font)
title.config(height=3)
title.pack(fill=X)

# Configuring main window background color
main.config(bg='grey')

# Button Frame
button_frame = Frame(main, bg='grey')
button_frame.pack(pady=20, padx=20)

# Buttons with improved appearance
style = ttk.Style()
style.configure('TButton', font=('Abadi Extra Light', 14, 'bold'), foreground='black', background='white')

upload_button = ttk.Button(button_frame, text="Upload Image", command=upload)
upload_button.pack(side=LEFT, padx=10)

process_button = ttk.Button(button_frame, text="Image Pre-Processing", command=imagepreprocess)
process_button.pack(side=LEFT, padx=10)

load_button = ttk.Button(button_frame, text="Model Load", command=load)
load_button.pack(side=LEFT, padx=10)

predict_button = ttk.Button(button_frame, text="Prediction", command=predict)
predict_button.pack(side=LEFT, padx=10)

# Text Display
font1 = ('Abadi Extra Light', 12)
text = Text(main, height=30, width=80)
text.pack(pady=20)
text.config(font=font1)
text.config(fg='black')  # Set text color to black
text.config(bg='white')

main.mainloop()
