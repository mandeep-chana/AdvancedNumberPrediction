import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna
import logging
import os
from datetime import datetime
import warnings
from optuna.logging import set_verbosity, WARNING
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings and set random seeds for reproducibility
warnings.filterwarnings('ignore')
set_verbosity(WARNING)
np.random.seed(42)
tf.random.set_seed(42)

# Configure logging and directories
log_dir = "ml_logs"
model_dir = "saved_models"
plots_dir = "analysis_plots"

for directory in [log_dir, model_dir, plots_dir]:
    os.makedirs(directory, exist_ok=True)

log_file = os.path.join(log_dir, f'prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class AdvancedNumberPredictor:
    def __init__(self, sequence_length=15):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.lstm_model = None
        self.rf_model = None
        self.gb_model = None
        self.et_model = None
        self.best_params = None
        self.number_frequencies = np.zeros(51)
        self.feature_names = None
        self.history = None

    def analyze_data_patterns(self, data):
        try:
            print("Analyzing data patterns...")
            self.number_frequencies = np.zeros(51)
            for num in data:
                if isinstance(num, (int, np.integer, float, np.floating)):
                    num_int = int(round(num))
                    if 1 <= num_int <= 50:
                        self.number_frequencies[num_int] += 1

            total_numbers = np.sum(self.number_frequencies)
            if total_numbers > 0:
                self.number_frequencies = self.number_frequencies / total_numbers

            logging.info("Data Analysis Complete")
            logging.info(f"Total numbers analyzed: {len(data)}")
            logging.info(f"Mean: {np.mean(data):.2f}")
            logging.info(f"Std: {np.std(data):.2f}")

            try:
                plt.figure(figsize=(15, 6))
                plt.bar(range(1, 51), self.number_frequencies[1:])
                plt.title('Number Frequency Distribution')
                plt.xlabel('Number')
                plt.ylabel('Frequency')
                plt.savefig(os.path.join(plots_dir, 'number_frequencies.png'))
                plt.close()
            except Exception as plot_error:
                logging.warning(f"Could not create plot: {plot_error}")

            return self.number_frequencies
        except Exception as e:
            logging.error(f"Error in data pattern analysis: {e}")
            self.number_frequencies = np.ones(51) / 50
            return self.number_frequencies

    def create_sequences(self, data):
        try:
            print("Creating sequences...")
            X, y = [], []

            valid_data = []
            for num in data:
                try:
                    num_float = float(num)
                    if 1 <= num_float <= 50:
                        valid_data.append(round(num_float))
                except (ValueError, TypeError):
                    continue

            valid_data = np.array(valid_data)

            if len(valid_data) <= self.sequence_length:
                raise ValueError(f"Not enough valid data points. Need more than {self.sequence_length} numbers.")

            for i in range(len(valid_data) - self.sequence_length):
                sequence = valid_data[i:(i + self.sequence_length)]
                target = valid_data[i + self.sequence_length]

                seq_features = [
                    np.mean(sequence),
                    np.std(sequence),
                    np.min(sequence),
                    np.max(sequence),
                    np.median(sequence),
                    np.percentile(sequence, 25),
                    np.percentile(sequence, 75),
                    self.number_frequencies[int(round(target))],
                    np.diff(sequence).mean(),
                    np.diff(sequence).std()
                ]

                X.append(np.concatenate([sequence, seq_features]))
                y.append(target)

            X = np.array(X)
            y = np.array(y)

            print(f"Created {len(X)} sequences")
            logging.info(f"Created {len(X)} sequences from {len(valid_data)} valid numbers")

            return X, y
        except Exception as e:
            logging.error(f"Error creating sequences: {e}")
            raise

    def optimize_hyperparameters(self, X_train, y_train):
        def objective(trial):
            print(f"\rTrial {trial.number + 1}/10", end='', flush=True)

            inputs = Input(shape=(X_train.shape[1], 1))

            x = Bidirectional(LSTM(
                trial.suggest_int('lstm_units', 32, 128)
            ))(inputs)
            x = BatchNormalization()(x)
            x = Dropout(trial.suggest_float('dropout1', 0.2, 0.4))(x)

            x = Dense(trial.suggest_int('dense_1', 32, 128), activation='relu')(x)
            x = Dropout(trial.suggest_float('dropout2', 0.1, 0.3))(x)

            outputs = Dense(1)(x)

            model = Model(inputs=inputs, outputs=outputs)

            learning_rate = trial.suggest_float('learning_rate', 1e-3, 1e-2, log=True)
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss='huber'
            )

            early_stopping = EarlyStopping(
                patience=5,
                restore_best_weights=True,
                monitor='val_loss'
            )

            history = model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=trial.suggest_categorical('batch_size', [32, 64, 128]),
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0
            )

            val_loss = min(history.history['val_loss'])
            return val_loss

        try:
            print("\n=== Starting Hyperparameter Optimization ===")
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=10)

            print("\n=== Optimization Complete ===")
            print("\nBest parameters found:")
            for key, value in study.best_params.items():
                print(f"{key}: {value}")

            self.best_params = study.best_params
            return study.best_params
        except Exception as e:
            logging.error(f"Error in hyperparameter optimization: {e}")
            raise
    def build_model(self, params):
        try:
            inputs = Input(shape=(self.sequence_length + 10, 1))

            x = Bidirectional(LSTM(params['lstm_units']))(inputs)
            x = BatchNormalization()(x)
            x = Dropout(params['dropout1'])(x)

            x = Dense(params['dense_1'], activation='relu')(x)
            x = Dropout(params['dropout2'])(x)

            outputs = Dense(1)(x)

            model = Model(inputs=inputs, outputs=outputs)
            model.compile(
                optimizer=Adam(learning_rate=params['learning_rate']),
                loss='huber'
            )
            return model
        except Exception as e:
            logging.error(f"Error building model: {e}")
            raise

    def fit(self, X, y):
        try:
            print("\n1. Preparing data...")
            X_scaled = self.scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

            print("\n2. Training ensemble models...")
            self.rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )

            self.gb_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )

            self.et_model = ExtraTreesRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )

            print("Training Random Forest...")
            self.rf_model.fit(X_train, y_train)
            print("Training Gradient Boosting...")
            self.gb_model.fit(X_train, y_train)
            print("Training Extra Trees...")
            self.et_model.fit(X_train, y_train)

            print("\n3. Optimizing and training LSTM model...")
            best_params = self.optimize_hyperparameters(X_train_reshaped, y_train)
            self.lstm_model = self.build_model(best_params)

            checkpoint = ModelCheckpoint(
                os.path.join(model_dir, 'best_model.keras'),
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            )

            history = self.lstm_model.fit(
                X_train_reshaped, y_train,
                epochs=100,
                batch_size=best_params['batch_size'],
                validation_split=0.2,
                callbacks=[
                    EarlyStopping(patience=10, restore_best_weights=True),
                    checkpoint
                ],
                verbose=1
            )

            predictions = {
                'lstm': self.lstm_model.predict(X_test_reshaped, verbose=0).flatten(),
                'rf': self.rf_model.predict(X_test),
                'gb': self.gb_model.predict(X_test),
                'et': self.et_model.predict(X_test)
            }

            print("\nModel Performance:")
            for name, pred in predictions.items():
                mae = mean_absolute_error(y_test, pred)
                mse = mean_squared_error(y_test, pred)
                print(f"{name.upper()} - MAE: {mae:.4f}, MSE: {mse:.4f}")

            self.history = history
            return history

        except Exception as e:
            logging.error(f"Error in training: {e}")
            raise

    def predict_next_numbers(self, last_sequence, num_predictions=5):
        try:
            predictions = []
            current_sequence = np.array(last_sequence[-self.sequence_length:])

            print("\nGenerating predictions...")
            for i in range(num_predictions):
                seq_features = [
                    np.mean(current_sequence),
                    np.std(current_sequence),
                    np.min(current_sequence),
                    np.max(current_sequence),
                    np.median(current_sequence),
                    np.percentile(current_sequence, 25),
                    np.percentile(current_sequence, 75),
                    np.mean(self.number_frequencies),
                    np.diff(current_sequence).mean(),
                    np.diff(current_sequence).std()
                ]

                full_sequence = np.concatenate([current_sequence, seq_features])
                scaled_sequence = self.scaler.transform(full_sequence.reshape(1, -1))

                lstm_pred = self.lstm_model.predict(scaled_sequence.reshape(1, -1, 1), verbose=0)[0][0]
                rf_pred = self.rf_model.predict(scaled_sequence)[0]
                gb_pred = self.gb_model.predict(scaled_sequence)[0]
                et_pred = self.et_model.predict(scaled_sequence)[0]

                ensemble_pred = (
                        0.4 * lstm_pred +
                        0.2 * rf_pred +
                        0.2 * gb_pred +
                        0.2 * et_pred
                )

                next_number = int(np.clip(round(ensemble_pred), 1, 50))
                predictions.append(next_number)

                current_sequence = np.append(current_sequence[1:], next_number)

                print(f"\nPrediction {i+1}:")
                print(f"  LSTM: {int(round(lstm_pred))}")
                print(f"  RF: {int(round(rf_pred))}")
                print(f"  GB: {int(round(gb_pred))}")
                print(f"  ET: {int(round(et_pred))}")
                print(f"  Ensemble: {next_number}")

            return np.array(predictions)

        except Exception as e:
            logging.error(f"Error in prediction: {e}")
            raise

def main():
    try:
        print("\n=== Advanced Number Prediction System ===")
        print("Loading data...")

        try:
            data = pd.read_csv('data.txt', sep='\t', header=None).values.flatten()
            print(f"Data loaded: {len(data)} numbers")
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Please ensure 'data.txt' exists and contains tab-separated numbers.")
            return

        predictor = AdvancedNumberPredictor(sequence_length=15)
        predictor.analyze_data_patterns(data)
        X, y = predictor.create_sequences(data)

        if len(X) == 0 or len(y) == 0:
            print("Error: No valid sequences could be created from the data.")
            return

        print("\nStarting model training...")
        predictor.fit(X, y)

        print("\nGenerating final predictions...")
        last_sequence = data[-predictor.sequence_length:]
        predictions = predictor.predict_next_numbers(last_sequence)

        print("\n=== Final Predictions ===")
        print("Next 5 numbers:", predictions.tolist())

        with open('predictions.txt', 'w') as f:
            f.write("Predicted numbers with details:\n")
            f.write("================================\n")
            for i, pred in enumerate(predictions, 1):
                f.write(f"Number {i}: {int(pred)}\n")
            f.write("\nPrediction generated at: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        print("\nPredictions saved to 'predictions.txt'")
        print("Analysis plots saved in 'analysis_plots' directory")
        print("Model checkpoints saved in 'saved_models' directory")

    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        print(f"\nError: {str(e)}")
        print("Check the log file for details.")

if __name__ == "__main__":
    main()

    