from flask import Flask, render_template, request
import os
import pandas as pd
import joblib
model = joblib.load('C:\\Users\\rosha\\Documents\\HTMLCSS\\static\\files\\svc_model.pkl')
app = Flask(__name__)

# Set a folder to save uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/signin')
def signin():
    return render_template('signin.html')

@app.route('/about-us')
def about_us():
    return render_template('about-us.html')
@app.route('/demo')
def demo():
    return render_template('demo.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    
    if 'fileInput' not in request.files:
        return "No file part", 400  # Bad Request

    file = request.files['fileInput']
    
    if file.filename == '':
        return "No selected file", 400  # Bad Request
    
    if file:
        # Create a secure filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Read the file into a DataFrame
        try:
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file.filename.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                return "Unsupported file type. Please upload a CSV or Excel file.", 400  # Bad Request
            
            try:
                data = {
                    'Addictive disorder': 0,
                    'Anxiety disorder': 1,
                    'Healthy control': 2,
                    'Mood disorder': 3,
                    'Obsessive compulsive disorder': 4,
                    'Schizophrenia': 5,
                    'Trauma and stress related disorder': 6
                }
                reverse_mapping = {v: k for k, v in data.items()}
                # Assuming df is preprocessed and suitable for the model
                predictions = model.predict(df)

                predicted_disorders = [reverse_mapping[pred] for pred in predictions]

                # Convert predictions to a DataFrame for display
                predictions_df = pd.DataFrame(predicted_disorders, columns=['Predicted Disorder'])
                issue =  predictions_df['Predicted Disorder'].iloc[0]
                return f'You are likely to have a {issue}'

            except Exception as e:
                return f"Error making predictions: {str(e)}", 500  # Internal Server Error

        except Exception as e:
            return f"Error reading file: {str(e)}", 500  # Internal Server Error



if __name__ == '__main__':
    app.run(debug=True)
