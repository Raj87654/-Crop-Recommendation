import os
import pickle
import pandas as pd

def load_model_and_encoder(filepath=None):
    if filepath is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, 'models', 'crop_model.pkl')
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['label_encoder'], data['feature_names']

def predict_crop(N, P, K, temperature, humidity, ph, rainfall, model, le, feature_names):
    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                              columns=feature_names)
    prediction_encoded = model.predict(input_data)
    predicted_crop = le.inverse_transform(prediction_encoded)
    return predicted_crop[0]

def main():
    try:
        model, le, feature_names = load_model_and_encoder()
    except FileNotFoundError:
        print("Model not found. Please run 'python train_model.py' first.")
        return

    print("Interactive Crop Recommendation Tool")
    print("-" * 36)
    
    while True:
        print("\nPlease enter the following values for crop recommendation:")
        try:
            input_N = float(input("Nitrogen (N): "))
            input_P = float(input("Phosphorous (P): "))
            input_K = float(input("Potassium (K): "))
            input_temperature = float(input("Temperature: "))
            input_humidity = float(input("Humidity: "))
            input_ph = float(input("pH: "))
            input_rainfall = float(input("Rainfall: "))

            predicted_crop = predict_crop(input_N, input_P, input_K, input_temperature, 
                                          input_humidity, input_ph, input_rainfall,
                                          model, le, feature_names)
            print(f"\nBased on your input, the recommended crop is: {predicted_crop.upper()}")

            continue_input = input("\nDo you want to predict another crop? (yes/no): ").lower()
            if continue_input not in ('yes', 'y'):
                print("Exiting crop recommendation tool. Goodbye!")
                break

        except ValueError:
            print("Invalid input. Please ensure all values are numbers.")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
