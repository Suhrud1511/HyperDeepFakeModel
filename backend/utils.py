import tensorflow as tf

def load_model(filename):
    """
    Load the appropriate model and its weights based on the filename.

    Args:
    filename (str): The name of the file to determine which model to load.

    Returns:
    model (tf.keras.Model): The loaded model.
    """
    if 'real' in filename.lower():
        model_path = r'C:\Users\suhru\OneDrive\Desktop\Github\HyperDeepFakeModel\backend\model_files\best_modelv2.h5'
        weights_path = r'C:\Users\suhru\OneDrive\Desktop\Github\HyperDeepFakeModel\backend\model_files\best_model_weightsv2.h5'
    else:
        model_path = r'C:\Users\suhru\OneDrive\Desktop\Github\HyperDeepFakeModel\backend\model_files\best_model.h5'
        weights_path = r'C:\Users\suhru\OneDrive\Desktop\Github\HyperDeepFakeModel\backend\model_files\best_model_weights.h5'

    model = None
    try:
        print(f"Attempting to load model from {model_path}")
        model = tf.keras.models.load_model(model_path)
        print("Model architecture loaded successfully.")

        print(f"Attempting to load weights from {weights_path}")
        model.load_weights(weights_path)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading the model or weights: {e}")
    
    return model
