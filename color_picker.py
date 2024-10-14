import is_it_red as iir
from tkinter import *

# importing the choosecolor package
from tkinter import colorchooser

# Function that will be invoked when the
# button will be clicked in the main window
def choose_color():

    # variable to store hexadecimal code of color
    color_code = colorchooser.askcolor(title ="Choose color") 
    text1.set(color_code[0])
    y_predict = iir.predict(iir.tree, iir.normalized_colour(color_code[0]))
    if y_predict:
        text2.set("True")
    else:
        text2.set("False")


root = Tk()

text1 = StringVar(root, "No color")
label1 = Label(root, textvariable=text1)
label1.pack()

text2 = StringVar(root, "None")
label2 = Label(root, textvariable=text2)
label2.pack()

button = Button(root, text = "Select color",
                command = choose_color)
button.pack()
root.geometry("300x300")
root.mainloop()