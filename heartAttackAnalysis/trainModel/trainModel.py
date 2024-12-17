import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential          # type: ignore
from tensorflow.keras.layers import Dense               # type: ignore
from tensorflow.keras.optimizers import Adam            # type: ignore


# Read csv file
df = pd.read_csv('heartAttackAnalysis/data/heart.csv')


# Encoding - We converted some data to numerical values.
label_encoder = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_cp = LabelEncoder()
df['Sex'] = label_encoder_sex.fit_transform(df['Sex'])
df['ChestPainType'] = label_encoder_cp.fit_transform(df['ChestPainType'])
df['RestingECG'] = label_encoder.fit_transform(df['RestingECG'])
df['ExerciseAngina'] = label_encoder.fit_transform(df['ExerciseAngina'])
df['ST_Slope'] = label_encoder.fit_transform(df['ST_Slope'])



# Define Input and Output
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']


# Training data and testing
X_train, X_test , y_train, y_text = train_test_split(X,y, test_size=0.3 , random_state=42)


# fix the data imbalance
smote = SMOTE(random_state=42)
X_train_smote , y_train_smote = smote.fit_resample(X_train, y_train)

# Scaling data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)


#1. ANN
print('Model 1: Artifical Neural Networks')
model = Sequential()
model.add(Dense(16, input_dim=X_train_scaled.shape[1] ,activation = 'relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy' , optimizer = optimizer , metrics=['accuracy'])

# Traning the model
model.fit(X_train_scaled, y_train_smote, epochs=100 , verbose=1 , validation_data=(X_test_scaled, y_text) )
loss, accuracy = model.evaluate(X_test_scaled, y_text)
print(f'Accuracy Rate: {accuracy * 100:.2f}%')



#2. Model - RANDOM FOREST
# print('Model 2: Random Forest')
# rf_model = RandomForestClassifier(random_state=42)
# rf_model.fit(X_train_smote, y_train_smote)
# y_pred_rf = rf_model.predict(X_test_scaled)
# rf_accuracy = accuracy_score(y_text, y_pred_rf)
# print(f'Accuracy rate : {rf_accuracy * 100:.2f}%')


# 3. DECISION TREE
# print('Model 3: Decision Tree')
# dt_model = DecisionTreeClassifier(random_state=42)
# dt_model.fit(X_train_smote , y_train_smote)
# y_pred_dt = dt_model.predict(X_test_scaled)
# dt_accuracy = accuracy_score(y_text , y_pred_dt)
# print(f'Accuracy rate: {dt_accuracy * 100:.2f}%')


#4.logistic Regression
# print('Model 4: Logistic Regression')
# lr_model = LogisticRegression(random_state=42)
# lr_model.fit(X_train_smote , y_train_smote)
# y_pred_lr = lr_model.predict(X_test_scaled)
# lr_accuracy = accuracy_score(y_text, y_pred_lr)
# print(f'Accuracy rate : {lr_accuracy * 100:.2f}%')



# Get input from users then make prediction
while True:
    user_input_1 = float(input('Age: '))
    user_input_2 = input('Sex (M/F): ')
    user_input_3 = input('Chest Pain Type (ATA, NAP, ASY, TA): ')
    user_input_4 = float(input('Resting BP: '))
    user_input_5 = float(input('Cholesterol: '))


    try:
        # Encoding
        user_data = pd.DataFrame({
            'Age': [user_input_1],
            'Sex': [label_encoder_sex.transform([user_input_2])[0]],            
            'ChestPainType': [label_encoder_cp.transform([user_input_3])[0]],
            'RestingBP': [user_input_4],
            'Cholesterol': [user_input_5],
            'FastingBS': [0],                           # fixed value
            'RestingECG': [0],                          # fixed value
            'MaxHR': [150],                             # fixed value
            'ExerciseAngina': [0],                      # fixed value
            'Oldpeak': [0.0],
            'ST_Slope': [1]
            })


        # Scaling the data
        user_data_scaled = scaler.transform(user_data)

        # Predict
        prediction = model.predict(user_data_scaled)
        print(f'Prediction result: Heart Disease rate: {prediction[0][0]:.4f}')


        if prediction >= 0.5:
            print('High risk of heart disease!')
        else:
            print('Low risk of heart disease')


    except ValueError as e:
        print(f"Something went wrong: {e}. Please check the information you entered.")


    # New prediction
    np = input('Do you want to make new prediction? (Y/N): ')
    if np.lower() != 'e':
        break