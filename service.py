import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import joblib  # joblib 추가
from sklearn.ensemble import RandomForestClassifier

# 모델 저장 경로 설정
model_path = 'basil_ginseng_classifier_model.pkl'

# GUI 설정
root = tk.Tk()
root.title('바질 , 인삼 분류기')

frame = tk.Frame(root)
frame.pack(pady=20)

image_label = tk.Label(frame)
image_label.pack()

result_label = tk.Label(frame, text='이미지를 업로드하여 분류 결과를 확인하세요.')
result_label.pack(pady=10)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (250, 250))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)  # 배열에서 이미지 객체로 변환
    img = ImageTk.PhotoImage(img)  # PIL 이미지를 Tkinter 이미지로 변환
    return img

def predict_image():
    file_path = filedialog.askopenfilename()  # 파일 대화상자로 이미지 파일 경로 선택
    if file_path:
        img = preprocess_image(file_path)
        image_label.config(image=img)
        image_label.image = img  # 이미지 레이블에 이미지 설정

        img = cv2.imread(file_path)
        img = cv2.resize(img, (150, 150))
        img = img.flatten() / 255.0
        img = img.reshape(1, -1)

        # 저장된 모델 로드
        rf_model = joblib.load(model_path)  # 저장된 모델 불러오기
        prediction = rf_model.predict(img)[0]
        result = '바질' if prediction == 0 else '인삼'
        result_label.config(text=f'예측 결과: {result}')

# 이미지 업로드 버튼
upload_button = tk.Button(frame, text='이미지 업로드', command=predict_image)
upload_button.pack()

root.mainloop()
