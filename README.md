# next-number-predictor
A Django-powered web application predicting the next number in a sequence using an LSTM Deep learning model. Deployed on an AWS EC2 instance running Ubuntu.

### Key Features

- **LSTM-based Model**: The project uses an LSTM model to predict the next number in a sequence.
- **Django Backend**: The application is powered by Django for web server management on which the whole project is made.
- **Interactive Web Interface**: Users can input their sequences through a simple form and receive the next number prediction and sequence type.
- **Model Saving**: The trained model is saved and loaded in .keras format for future use without retraining.

  ## Technologies Used

- **Python**: Primary language for both backend and machine learning parts.
- **Django**: Web framework for building the server and API to interact with users.
- **TensorFlow**: Machine learning library for building the LSTM model.
- **HTML/CSS**: For creating the frontend interface for the application.
- **AWS**: The backend server is based on ec2 instance from aws.
