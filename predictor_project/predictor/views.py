import json
import os
import logging
import numpy as np
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST
from predictor.predictor import ImprovedSequencePredictor

# Set the path to your .keras model file
model_path = os.path.join(os.path.dirname(__file__), "predictor_model.keras")

logger = logging.getLogger(__name__)

# Create a global instance of the predictor
predictor = None

def load_model():
    global predictor
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        predictor = ImprovedSequencePredictor(model_path)
        logger.debug(f"Predictor initialized: {predictor}")
        logger.debug(f"Available methods: {dir(predictor)}")
        print("Model Loaded Suc.")

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")


# Ensure the model is loaded at startup
load_model()

def validate_sequence(sequence):
    """
    Validate the input sequence to ensure it is in the correct format.
    """
    if not sequence:
        raise ValueError("The input sequence is empty.")
    if not isinstance(sequence, list):
        raise ValueError("The input sequence must be a list.")
    validated_sequence = []
    for num in sequence:
        try:
            # Convert numeric strings to float
            validated_sequence.append(float(num))
        except (ValueError, TypeError):
            raise ValueError(f"Invalid value in sequence: {num}")
    return validated_sequence


def index(request):
    """
    Render the index page for the predictor.
    """
    return render(request, 'predictor/index.html')

@require_POST
def predict(request):
    """Handle prediction requests."""
    try:
        # Parse the JSON body to extract the input sequence
        data = json.loads(request.body)
        sequence = data.get('sequence', [])

        print(sequence)
        print("----")
        # Validate and preprocess the sequence
        if not sequence:
            return JsonResponse({'status': 'error', 'message': 'No input sequence provided'}, status=400)
        print("+++++")
        # Convert string numbers to floats, handling potential conversion issues
        try:
            sequence = [float(num) for num in sequence]
        except ValueError:
            return JsonResponse({'status': 'error', 'message': 'Invalid number in sequence'}, status=400)
        
        print('1')

        # Add this for model state check
        if predictor is None or predictor.model is None:
            logger.error("Predictor or model not properly initialized")
            load_model()
            if predictor is None or predictor.model is None:
                return JsonResponse({'status': 'error', 'message': 'Model initialization failed'}, status=500)
            
        print('2')

        # Add pre-validation debug logs
        logger.debug(f"Pre-validation sequence: {sequence}")
        sequence = validate_sequence(sequence)
        logger.debug(f"Post-validation sequence: {sequence}")

        print('2.5')

        # Detect pattern type and verify logic
        pattern_type = predictor.detect_pattern(sequence)
        logger.debug(f"Detected pattern type: {pattern_type}")

        print('3')

        #if pattern_type == 'arithmetic':
        #    # Verify arithmetic calculation
        #    diff = sequence[1] - sequence[0]
        #    expected = sequence[-1] + diff
        #    logger.debug(f"Arithmetic check - diff: {diff}, expected: {expected}")

        # Make the prediction
        predictor.fit(sequence)
        prediction = predictor.predict_next(sequence)

        print(prediction)
        print('----------------')

        # Handle the prediction result
        if isinstance(prediction, (np.ndarray, np.float32, np.float64)):
            prediction = prediction.tolist() if isinstance(prediction, np.ndarray) else float(prediction)

        # Return the prediction and pattern type in the response
        return JsonResponse({
            'status': 'success',
            'prediction': prediction,
            'pattern_type': pattern_type
        })
    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Invalid JSON'}, status=400)
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


def predict_view(sequence):
    """
    Predict the next value given a sequence.
    """
    try:
        logger.debug(f"Received sequence for prediction: {sequence}")
        prediction = predictor.predict_next(sequence)
        logger.debug(f"Prediction result: {prediction}")
        return prediction
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return None

