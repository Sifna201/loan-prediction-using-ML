# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 21:34:52 2021

@author: user
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
import tkinter
from tkinter import *

window=tkinter.Tk()
window.title('loan prediction')
window.geometry("600x400")
window.config(bg='pink')
#id=e1.get()
#type(id)
def function():
    
    testdata=joblib.load('dataf.xml')
    dect=joblib.load('dect.xml')
    id=e1.get()
    dect=joblib.load('dect.xml')
    testdata=np.array(testdata)
    testdata=testdata[int(id)].reshape(1,-1)
    print(testdata)
    y_pred=dect.predict(testdata)
    print(y_pred)
    window2 = Toplevel(window)
    window2.title('Result')
    window2.geometry("600x400")
    window2.config(bg='sky blue')
    e2=Entry( window2,width=40)
    e2.pack()
    label2=tkinter.Label(window2,text='RESULT',font=('arial',40)).pack()
    Button (window2,text="Exit",command=window.destroy,bg="yellow",fg="black",activebackground = "pink",width=10,height=3).pack(side="bottom")

    Button( window2,text="Back",command=window2.destroy,bg="pink",fg="black",activebackground = "pink",width=10,height=3).pack(side="bottom")
    if y_pred==0:
        text=("Loan Cannot Be Approved")
        e2.insert(1,text)
        
    else:
        text=("Loan Can Be Approved")
        e2.insert(1,text)
    return

#Num =tkinter.Label(window, text="Enter The no",bg="white",fg="black",width=30,height=1).pack()
label=tkinter.Label(window,text='LOAN PREDICTION',font=('arial',40)).pack()

e1=Entry(window,width=70,bg='orange')

e1.pack()

Button(window,text="enter",command=function,bg="blue",fg="black",width=10,height=4).pack(side="bottom")

    
window.mainloop()