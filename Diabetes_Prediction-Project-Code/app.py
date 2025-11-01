from flask import Flask,request, url_for, redirect, render_template  ## importing necessary libraries
import pickle  ## pickle for loading model(Diabetes.pkl)
import pandas as pd  ## to convert the input data into a dataframe for giving as a input to the model

app = Flask(__name__)  ## setting up flask name

model = pickle.load(open("Diabetes.pkl", "rb"))  ##loading model


@app.route('/')             ## Defining main index route
def home():
    return render_template("index.html")   ## showing index.html as homepage


@app.route('/predict',methods=['POST','GET'])  ## this route will be called when predict button is called
def predict(): 
    #int_features=[float(x) for x in request.form.values()]
    # Get input values and convert to float
    try:
        values = []
        for i in range(1, 9):
            val = request.form[str(i)].strip()
            if not val:  # Handle empty values
                return render_template('index.html', pred='Please fill in all fields')
            values.append(float(val))
        
        row_df = pd.DataFrame([values], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                                               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    except ValueError:
        return render_template('index.html', pred='Please enter valid numeric values for all fields')
    print(row_df)
    try:
        prediction = model.predict_proba(row_df)  # Predicting the output
        prob = float(prediction[0][1])
        output = f"{prob:.2f}"
    except Exception as e:
        # return helpful error on template if prediction fails
        return render_template('index.html', pred=f'Prediction error: {e}')

    # Numeric comparison (not string) and friendly messages
    if prob > 0.5:
        return render_template('index.html', pred=f'Your chance of having diabetes is high ({output})')
    else:
        return render_template('index.html', pred=f'You are likely safe. Probability of having diabetes: {output}')




if __name__ == '__main__':
    print("Starting the Flask application...")
    app.run(host='127.0.0.1', port=8080, debug=True)          ## Running the app as debug==True