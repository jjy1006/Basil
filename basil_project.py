import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import joblib  # joblib 추가
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 데이터 경로 설정
base_dir = 'dataset'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# 모델 저장 경로 설정
model_path = 'basil_ginseng_classifier_model.pkl'

# 훈련 데이터 및 검증 데이터 로딩 및 전처리 함수
def load_and_preprocess_data(directory):
    X = []
    y = []
    num_images = 0
    for class_name, label in zip(['basil', 'ginseng'], [0, 1]):
        class_dir = os.path.join(directory, class_name)
        num_images += len(os.listdir(class_dir))
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            img = cv2.imread(image_path)  # 이미지 읽기
            img = cv2.resize(img, (150, 150))  # 이미지 크기 조정
            img = img.flatten() / 255.0  # 이미지 스케일링 및 1차원 벡터로 변환
            X.append(img)
            y.append(label)
    return np.array(X), np.array(y), num_images

# 훈련 데이터 및 검증 데이터 로딩
X_train, y_train, num_train_images = load_and_preprocess_data(train_dir)
X_val, y_val, num_val_images = load_and_preprocess_data(validation_dir)

# Random Forest 모델 학습
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 모델 저장
joblib.dump(rf_model, model_path)
print(f"모델이 {model_path}에 저장되었습니다.")

# 정확도 평가
y_pred_train = rf_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred_train)
print(f"Train Accuracy: {train_accuracy*100:.2f}%")

y_pred_val = rf_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_pred_val)
print(f"Validation Accuracy: {val_accuracy*100:.2f}%")

# GUI 설정
root = tk.Tk()
root.title('바질 또는 인삼 분류기')

frame = tk.Frame(root)
frame.pack(pady=20)

image_label = tk.Label(frame)
image_label.pack()

result_label = tk.Label(frame, text=f'훈련 데이터셋 이미지 수: {num_train_images}, 검증 데이터셋 이미지 수: {num_val_images}')
result_label.pack(pady=10)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (250, 250))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)  # 배열에서 이미지 객체로 변환
    img = ImageTk.PhotoImage(img)  # PIL 이미지를 Tkinter 이미지로 변환
    return img

def predict_image():
    file_path = filedialog.askopenfilename()  
    if file_path:
        img = preprocess_image(file_path)
        image_label.config(image=img)
        image_label.image = img  # 이미지 레이블에 이미지 설정
        
        img = cv2.imread(file_path)
        img = cv2.resize(img, (150, 150))
        img = img.flatten() / 255.0
        img = img.reshape(1, -1)
        
        # 모델 예측
        prediction = rf_model.predict(img)[0]
        result = '바질' if prediction == 0 else '인삼'
        result_label.config(text=f'예측 결과: {result}')

upload_button = tk.Button(frame, text='이미지 업로드', command=predict_image)
upload_button.pack()

root.mainloop()
