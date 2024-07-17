import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler

# 저장된 벡터와 레이블 불러오기
data_path = '/home/hui/emotion_detect/src/asian_train_copy/get_only_asian_dataset/data.npy'
labels_path = '/home/hui/emotion_detect/src/asian_train_copy/get_only_asian_dataset/label.npy'
data = np.load(data_path)
labels = np.load(labels_path)

# 레이블 이진화
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# 데이터 불균형 문제 해결을 위해 오버샘플링
ros = RandomOverSampler(random_state=42)
data, labels = ros.fit_resample(data, labels)

# 데이터를 훈련, 검증, 테스트 세트로 분할 (70%, 15%, 15%)
train_data, temp_data, train_labels, temp_labels = train_test_split(data, labels, test_size=0.3, random_state=42)
validation_data, test_data, validation_labels, test_labels = train_test_split(temp_data, temp_labels, test_size=0.5, random_state=42)  # 0.5 x 0.3 = 0.15

# 데이터셋 분할 후 크기 출력
print(f"Train data shape: {train_data.shape}")
print(f"Validation data shape: {validation_data.shape}")
print(f"Test data shape: {test_data.shape}")

# 모델 구축
model = Sequential([
    Dense(256, input_shape=(512,), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.7),
    Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.7),
    Dense(labels.shape[1], activation='softmax')  # 클래스 수에 따라 변경 (여기서는 happy, sad, angry)
])

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
history = model.fit(
    train_data, train_labels,
    epochs=50,
    batch_size=128,
    validation_data=(validation_data, validation_labels),
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
)

# 모델 평가
test_loss, test_acc = model.evaluate(test_data, test_labels)
print(f'Test accuracy: {test_acc}')

# 모델 저장
model.save('Asian_emotion_ArcFace_model.h5')

# 학습 결과 시각화
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
