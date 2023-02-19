from tkinter import *
import ctypes
import pandas
import string
from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn import feature_extraction
from sklearn import pipeline
from sklearn import linear_model
from tkinter import filedialog
import threading





user32 = ctypes.windll.user32
screensize = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
root = Tk()
width_screen = int(int(screensize[0])//2)
height_screen = int(int(screensize[1])//2)
root.geometry(f"{width_screen}x{height_screen}")
root.title("Language Detector Using ML~HauseMaster")


text_field = Text(root, fg="black", highlightthickness="1", height=2, width=width_screen//12, bg="yellow")

text_field.place(x=60, y=40)

text_field.insert("1.0", "Enter Any Text...")


def punctuationRemove(text):
    for i in string.punctuation:
        text = text.replace(i, "")
    text = text.lower()
    return text


model_arr = []


def train():
    global model_arr, dataset_inp, wait
    dataset_path = dataset_inp.get("1.0", END).strip()
    wait.config(text="Pleast Wait...")
    wait.place(x=285, y=270)
    data = pandas.read_csv(dataset_path)
    data['Text'] = data['Text'].apply(punctuationRemove)
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .2)
    vec = feature_extraction.text.TfidfVectorizer(ngram_range = (1, 3), analyzer='char')
    main_model = pipeline.Pipeline([('vec', vec), ('clf', linear_model.LogisticRegression())])
    main_model.fit(x_train, y_train)
    prediction = main_model.predict(x_test)
    wait.config(text="Training Complete")
    model_arr.append(main_model)


def translate():
    global text_field, language_result, model_arr
    txt = text_field.get("1.0", END)
    main_model = model_arr[-1]
    # main_model = train()
    language_result.config(text=f"Detected Language is: {main_model.predict([txt])[0]}")


def openFilePicker():
    global dataset_inp
    dataset_path = filedialog.askopenfilename()
    dataset_inp.delete("1.0", END)
    dataset_inp.insert("1.0", dataset_path)


detect_button = Button(root, text="Detect Language", command=translate, padx=6, pady=10)
detect_button.place(x=550, y=35)

language_result = Label(root, text="Detected Language is: ")
language_result.place(x=250, y=100)

dataset_btn = Button(root, text="Select Dataset File", command= openFilePicker)
dataset_btn.place(x = 10, y = 200)
 
dataset_inp = Text(root, fg="black", highlightthickness="1", height=1, width=width_screen//10)
dataset_inp.place(x=120, y=200)

train_btn = Button(root, text="Start Training", command=threading.Thread(target=train).start)
train_btn.place(x=280, y= 240)

wait = Label(root, text="Please Wait...")
root.mainloop()


