from tkinter import *
import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
import numpy as np
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf


def display_page():
    login_window.destroy()  # type: ignore


display_window = Tk()
display_window.geometry('990x660+50+50')
display_window.resizable(0, 0)
display_window.title('Before Implementation')
background = ImageTk.PhotoImage(file='bg5.png')
bgLabel = Label(display_window, image=background)
bgLabel.place(x=0, y=0)

headingButton = Button(text='PanCard Fraud Detection', font=('Albertus Extra Bold', 20, 'bold'), bd=0, bg='white',
                       fg='firebrick1', activebackground='white', activeforeground='firebrick1', width=60, height=1)
headingButton.grid(row=0, column=0, pady=40)

frame = Frame(display_window, bg='white')
frame.place(x=50, y=140)
img_path = ""


def set_img_path(img_path):
    return img_path


def openimage():
    fileName = askopenfilename(initialdir='', title='Select image for Aanalysis ',
                               filetypes=[("all files", "*.*")])
    imgpath = fileName
    global global_path
    global_path = imgpath

    img = Image.open(imgpath)
    img = np.array(img)

    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(im)
    img = tk.Label(display_window, image=imgtk, height=480, width=600)

    img.image = imgtk
    img.place(x=370, y=130)


def predict():
    model = tf.keras.models.load_model("pan_tamper_detection_model.h5")

    test_images = []
    image = tf.keras.preprocessing.image.load_img(global_path,
                                                  target_size=(150, 150))
    image = tf.keras.preprocessing.image.img_to_array(image)
    test_images.append(image)

    test_images = np.array(test_images) / 255.0  # Normalize pixel values

    predictions = model.predict(test_images)

    if predictions[0] < 0.5:
        result = "Not Tampered"
    else:
        result = "Tampered"

    messagebox.showinfo("PAN Card Tamper Detection", f"The PAN card is {result}")


imageButton = Button(frame, text='Browse Image', font=('Open Sans', 16), bd=0, bg='firebrick1',
                     fg='white', activebackground='firebrick1', activeforeground='white', width=20, cursor='hand2',
                     command=openimage)
imageButton.grid(row=1, column=0, pady=50)

# imagepreprocessButton=Button(frame,text='Image_Preprocess',font=('Open Sans',16),bd=0,bg='firebrick1',
#                     fg='white',activebackground='firebrick1',activeforeground='white',width=20,cursor='hand2',command=convert_grey)
# imagepreprocessButton.grid(row=2,column=0,pady=40)

predictionButton = Button(frame, text='CNN-Prediction', font=('Open Sans', 16), bd=0, bg='firebrick1',
                          fg='white', activebackground='firebrick1', activeforeground='white', width=20, cursor='hand2',
                          command=predict)
predictionButton.grid(row=3, column=0, pady=50)

exitButton = Button(frame, text='EXIT', font=('Open Sans', 16), bd=0, bg='firebrick1',
                    fg='white', activebackground='firebrick1', activeforeground='white', width=20, cursor='hand2',
                    command=exit)
exitButton.grid(row=4, column=0, pady=50)

displayLabel = tk.LabelFrame(frame, text='Process:-', font=('Open Sans', 9), fg='black', bg='white', width=20)
displayLabel.grid(row=0, padx=12, column=0)

display_window.mainloop()