import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Conv2D, 
                                   MaxPooling2D, Flatten, Concatenate)
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import backend as K
import joblib
import os

# ----------------------------
# Step 1: Generate Dummy ICU Data
# ----------------------------
def generate_dummy_data(num_samples=1000):
    np.random.seed(42)
    
    data = {
        "heart_rate": np.concatenate([
            np.random.normal(80, 10, int(num_samples*0.9)),
            np.random.normal(120, 15, int(num_samples*0.1))
        ]),
        "respiratory_rate": np.concatenate([
            np.random.normal(18, 3, int(num_samples*0.85)),
            np.random.normal(30, 5, int(num_samples*0.15))
        ]),
        "spo2": np.concatenate([
            np.random.normal(96, 2, int(num_samples*0.9)),
            np.random.normal(85, 3, int(num_samples*0.1))
        ]),
        "creatinine": np.concatenate([
            np.random.normal(1.2, 0.5, int(num_samples * 0.7)),
            np.random.lognormal(1.1, 0.4, int(num_samples * 0.3))  # More high values
        ]),
        "bun": np.random.normal(15, 4, num_samples),
        "alt": np.random.normal(30, 10, num_samples),
        "ast": np.random.normal(32, 12, num_samples),
        "sodium": np.random.normal(140, 5, num_samples),
        "potassium": np.random.normal(4.2, 0.5, num_samples),
        "calcium": np.random.normal(9.5, 0.5, num_samples),
        "lactate": np.concatenate([
            np.random.normal(1.5, 0.6, int(num_samples*0.85)),
            np.random.normal(4.0, 1.0, int(num_samples*0.15))
        ]),
        "coagulation_profile": np.random.normal(1.1, 0.2, num_samples),
        "blood_pressure": np.random.normal(120, 15, num_samples),
        "temperature": np.random.normal(37, 0.5, num_samples),
        "urine_output": np.concatenate([
            np.random.normal(50, 10, int(num_samples*0.9)),
            np.random.normal(20, 5, int(num_samples*0.1))
        ]),
        "glasgow_coma_scale": np.concatenate([
            np.random.normal(14, 1.5, int(num_samples*0.9)),
            np.random.normal(8, 2.0, int(num_samples*0.1))
        ]),
        "image_path": [f"fake_scan_{i}.png" for i in range(num_samples)]
    }
    
    df = pd.DataFrame(data)
    df["risk"] = (
        (df["creatinine"] > 1.5).astype(int) * 0.4 +
        (df["spo2"] < 92).astype(int) * 0.3 +
        (df["glasgow_coma_scale"] < 12).astype(int) * 0.3
    )
    # Convert to binary risk: 1 if score > 0.5 else 0
    df["risk"] = (df["risk"] > 0.5).astype(int)

    # 🚨 Force High Risk if creatinine > 5
    df.loc[df["creatinine"] > 5, "risk"] = 1

    return df

# ======================
# 2. NUMERICAL MODEL 
# ======================
def build_numeric_model(input_shape):
    inputs = Input(shape=input_shape)

    x = Dense(64, activation='relu')(inputs)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs)

# ======================
# 3. IMAGE MODEL
# ====================== 
def build_image_model(input_shape=(256, 256, 3)):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3,3), activation='relu')(inputs)
    x = MaxPooling2D(2,2)(x)
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPooling2D(2,2)(x)
    x = Flatten()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs)
# ======================
# 4. TRAINING PIPELINE
# ======================
def custom_weighted_loss(y_true, y_pred):
    creatinine = y_true[:, 1]  # Assuming it's 2nd column in stacked labels
    penalty = tf.where(creatinine > 5, 5.0, 1.0)
    bce = tf.keras.losses.binary_crossentropy(y_true[:, 0], y_pred)
    return tf.reduce_mean(bce * penalty)

def train_models():
    # Prepare data
    df = generate_dummy_data()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
    # Numerical data
    scaler = StandardScaler()
    num_features = [col for col in df.columns if col not in ["risk", "image_path"]]
    X_train_num = scaler.fit_transform(train_df[num_features])
    X_test_num = scaler.transform(test_df[num_features])
    joblib.dump(scaler, "scaler.save")
    

    # Image data (dummy)
    X_train_img = np.random.rand(len(train_df), 256, 256, 3)
    X_test_img = np.random.rand(len(test_df), 256, 256, 3)

    # Targets
    y_train = train_df["risk"].values.reshape(-1, 1)
    y_test = test_df["risk"].values.reshape(-1, 1)
    # Stack risk + creatinine for custom loss
    y_train_stacked = np.column_stack([y_train, X_train_num[:, 3]])  # index 3 = creatinine
    y_test_stacked = np.column_stack([y_test, X_test_num[:, 3]])

    # Train numerical model
    numeric_model = build_numeric_model(X_train_num.shape[1:])
    numeric_model.compile(optimizer=Adam(0.001), loss=custom_weighted_loss, metrics=['accuracy'])
    numeric_model.fit(X_train_num, y_train_stacked, epochs=10, validation_data=(X_test_num, y_test_stacked))

    # Train image model 
    image_model = build_image_model()
    image_model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    image_model.fit(X_train_img, y_train, epochs=5, validation_data=(X_test_img, y_test))

    # Save models
    numeric_model.save("numeric_model.h5")
    image_model.save("image_model.h5")
    print("✅ Both models trained and saved!")

# ======================
# 5. PREDICTION PIPELINE 
# ======================
class LifeGuardPredictor:
    def __init__(self):
        self.numeric_model = load_model("numeric_model.h5")
        self.image_model = load_model("image_model.h5")
        self.scaler = joblib.load("scaler.save")
    
    def predict(self, numeric_data, image=None):
        # Scale numerical data
        numeric_data = self.scaler.transform([numeric_data])

        # Get predictions
        num_pred = self.numeric_model.predict(numeric_data)[0][0]

        if image is not None:
            img_pred = self.image_model.predict(np.expand_dims(image, 0))[0][0]
            combined = (num_pred * 0.7) + (img_pred * 0.3)  # Weighted combination
            return {
                "numeric_risk": float(num_pred),
                "image_risk": float(img_pred),
                "combined_risk": float(combined)
            }
        return {"numeric_risk": float(num_pred)}
# Run training
if __name__ == "__main__":
    train_models()

