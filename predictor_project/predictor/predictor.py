#from google.colab import drive
#drive.mount('/content/drive' , force_remount=True)

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
import numpy.polynomial.polynomial as poly
import os


class ImprovedSequencePredictor:
    def __init__(self, min_sequence_length=2):
        self.min_sequence_length = 2
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.pattern_type = None
        self.model = None
        self.pattern_params = {}
        
        # Create the directory if it doesn't exist
        self.model_save_path = r"/home/ubuntu/final_project/predictor_project/predictor"
        os.makedirs(self.model_save_path, exist_ok=True)
        
        self.model_path = os.path.join(self.model_save_path, "predictor_model.keras")
        
        # Only attempt to load if the file exists
        if os.path.exists(self.model_path):
            self.model = self.load_model()
        else:
            print(f"No existing model found at {self.model_path}")

        
    def detect_pattern(self, sequence):
        """Detect the type of pattern in the sequence with improved accuracy"""
        if len(sequence) < self.min_sequence_length:
            return 'unknown'

        # Convert sequence to numpy array and handle any string inputs
        sequence = np.array(sequence, dtype=float)

        # Check for arithmetic sequence - made more robust
        differences = np.diff(sequence)
        if np.allclose(differences, differences[0], rtol=1e-5, atol=1e-5):
            self.pattern_params['difference'] = float(differences[0])
            return 'arithmetic'

        # Check for geometric sequence - added error handling for zero values
        try:
            ratios = sequence[1:] / sequence[:-1]
            if np.allclose(ratios, ratios[0], rtol=1e-5, atol=1e-5):
                self.pattern_params['ratio'] = float(ratios[0])
                return 'geometric'
        except:
            pass

        # Check for polynomial pattern (quadratic) - improved MSE threshold
        try:
            x = np.arange(len(sequence))
            coeffs = poly.polyfit(x, sequence, 2)
            y_pred = poly.polyval(x, coeffs)
            mse = np.mean((sequence - y_pred) ** 2)
            if mse < 0.05 * np.var(sequence):  # Made threshold stricter
                self.pattern_params['coefficients'] = coeffs
                return 'polynomial'
        except:
            pass

        # Check for Fibonacci-like sequence
        if self.is_fibonacci_like(sequence):
            return 'fibonacci'

        # Only use complex/neural network for sequences that don't fit other patterns
        if len(sequence) >= 5:  # Require more data points for complex pattern
            return 'complex'
            
        return 'unknown'

    def is_fibonacci_like(self, sequence):
        """Check if sequence follows Fibonacci-like pattern (each number is sum of previous two)"""
        if len(sequence) < 3:
            return False

        try:
            for i in range(2, len(sequence)):
                if not np.isclose(sequence[i], sequence[i-1] + sequence[i-2], rtol=1e-3, atol=1e-3):
                    return False
            return True
        except:
            return False

    def build_neural_model(self, lookback):
        """Build neural model for complex patterns with improved architecture"""
        model = models.Sequential([
            layers.Input(shape=(lookback, 1)),
            layers.LSTM(128, return_sequences=True, activation='relu'),
            layers.Dropout(0.2),
            layers.LSTM(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model

    def is_arithmetic(self, sequence):
        diff = sequence[1] - sequence[0]
        return all(sequence[i] - sequence[i - 1] == diff for i in range(2, len(sequence)))

    def predict_arithmetic(self, sequence):
        diff = sequence[1] - sequence[0]
        return sequence[-1] + diff

    def is_geometric(self, sequence):
        ratio = sequence[1] / sequence[0]
        return all(sequence[i] / sequence[i - 1] == ratio for i in range(2, len(sequence)))

    def predict_geometric(self, sequence):
        ratio = sequence[1] / sequence[0]
        return sequence[-1] * ratio

    def is_polynomial(self, sequence):
        # Simplified for demonstration; real polynomial detection is complex
        if len(sequence) < 3:
            return False
        return sequence[1] - sequence[0] == sequence[2] - sequence[1]

    def predict_polynomial(self, sequence):
        if len(sequence) < 3:
            raise ValueError("At least 3 numbers are required to predict polynomial")
        
        return sequence[-1] + (sequence[-1] - sequence[-2])  # Linear approximation

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model

    def fit(self, sequence):
        """Fit the predictor to the sequence with improved error handling"""
        try:
            print("********************")
            print(type(len(sequence)) , type(self.min_sequence_length))
            print('*****************')
            if len(sequence) < self.min_sequence_length:
                raise ValueError(f"Sequence must have at least {self.min_sequence_length} numbers")

            sequence = np.array(sequence, dtype=float)
            self.pattern_type = self.detect_pattern(sequence)

            if self.pattern_type == 'complex':
                lookback = min(len(sequence) - 1, max(3, len(sequence) // 2))
                scaled_sequence = self.scaler.fit_transform(sequence.reshape(-1, 1))

                X, y = [], []
                for i in range(len(scaled_sequence) - lookback):
                    X.append(scaled_sequence[i:(i + lookback)])
                    y.append(scaled_sequence[i + lookback])

                X = np.array(X)
                y = np.array(y)

                # Add early stopping and increase epochs
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='loss', patience=10, restore_best_weights=True)
                
                self.model = self.build_neural_model(lookback)
                self.model.fit(X, y, epochs=200, verbose=0, callbacks=[early_stopping])

            return self
        except Exception as e:
            print(f"Error in fit: {str(e)}")
            self.pattern_type = 'unknown'
            return self

    def predict_next(self, sequence):
        """Predict the next number in the sequence with improved error handling"""
        try:
            sequence = np.array(sequence, dtype=float)
            
            if self.pattern_type == 'arithmetic':
                return self.predict_arithmetic(sequence)
            elif self.pattern_type == 'geometric':
                return self.predict_geometric(sequence)
            elif self.pattern_type == 'polynomial':
                return self.predict_polynomial(sequence)
            elif self.pattern_type == 'fibonacci':
                return self.predict_fibonacci(sequence)
            elif self.pattern_type == 'complex' and self.model is not None:
                scaled_sequence = self.scaler.transform(sequence.reshape(-1, 1))
                lookback = self.model.input_shape[1]
                X = scaled_sequence[-lookback:].reshape(1, lookback, 1)
                scaled_prediction = self.model.predict(X, verbose=0)
                prediction = self.scaler.inverse_transform(scaled_prediction)
                return float(prediction[0][0])
            else:
                # For unknown patterns, use simple linear extrapolation
                if len(sequence) >= 2:
                    return float(sequence[-1] + (sequence[-1] - sequence[-2]))
                return float(sequence[-1])
                
        except Exception as e:
            print(f"Error in predict_next: {str(e)}")
            if len(sequence) > 0:
                return float(sequence[-1])  # Return last number if prediction fails
            return 0.0
    
    def save_model(self):
        """Save the model with proper error handling and checks"""
        if self.model is not None:
            try:
                # Ensure directory exists
                os.makedirs(self.model_save_path, exist_ok=True)
                
                # Save the model
                self.model.save(self.model_path)
                print(f"Model successfully saved to {self.model_path}")
                
                # Verify the file was created
                if os.path.exists(self.model_path):
                    print(f"Verified: File exists at {self.model_path}")
                else:
                    print("Warning: File was not created successfully")
                    
            except Exception as e:
                print(f"Error saving model: {str(e)}")
        else:
            print("No model to save - train the model first using the fit() method")

    def load_model(self):
        """Load the model with proper error handling"""
        try:
            if not os.path.exists(self.model_path):
                print(f"No model file found at {self.model_path}")
                return None
                
            model = tf.keras.models.load_model(self.model_path)
            print(f"Model successfully loaded from {self.model_path}")
            return model
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None