import os
import pickle
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from utils import load_data, preprocess_data, split_data

def main():
    # Construct paths dynamically based on script location
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print("Loading data...")
    data_path = os.path.join(base_dir, 'data', 'Crop_recommendation.csv')
    df = load_data(data_path)
    
    # Save crop distribution plot
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='label', palette='viridis', hue='label', legend=False)
    plt.title('Crop Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    dist_plot_path = os.path.join(base_dir, 'images', 'crop_distribution.png')
    plt.savefig(dist_plot_path)
    print(f"Saved crop distribution plot to {dist_plot_path}")
    
    print("Preprocessing data...")
    X, y_encoded, le = preprocess_data(df)
    
    # Save the feature names and label encoder for prediction
    feature_names = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = split_data(X, y_encoded)
    
    print("Training XGBoost model...")
    model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(le.classes_), random_state=42)
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy Score: {accuracy:.4f}")
    
    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    cm_plot_path = os.path.join(base_dir, 'images', 'confusion_matrix.png')
    plt.savefig(cm_plot_path)
    print(f"Saved confusion matrix plot to {cm_plot_path}")
    
    # Save the model and label encoder
    print("Saving model and label encoder...")
    model_path = os.path.join(base_dir, 'models', 'crop_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'label_encoder': le,
            'feature_names': feature_names
        }, f)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
