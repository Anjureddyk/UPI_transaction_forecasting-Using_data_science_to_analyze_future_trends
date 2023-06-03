from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained Lasso model
with open('lasso_model.pkl', 'rb') as file:
    lasso_model = pickle.load(file)

# Load the label encoder
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# Load the dataset
data = pd.read_csv("preprocessed_data.csv")

# Check if all required columns exist in the dataset
required_columns = ['UPI_Banks', 'Volume_Mn_By_Customers', 'Volume_Mn', 'Value_Cr', 'Month', 'Year', 'Total_Volume_Mn']
for column in required_columns:
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in the dataset.")

# Extract the unique UPI bank names
upi_banks = data['UPI_Banks'].unique().tolist()

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        upi_bank = request.form['upi_bank']
        volume_customers = float(request.form['volume_customers'])
        volume = float(request.form['volume'])
        value = float(request.form['value'])
        month = int(request.form['month'])
        year = int(request.form['year'])
        total_volume = float(request.form['total_volume'])
        
        # Create a DataFrame with the input values
        input_data = pd.DataFrame({
            'UPI_Banks': [upi_bank],
            'Volume_Mn_By_Customers': [volume_customers],
            'Volume_Mn': [volume],
            'Value_Cr': [value],
            'Month': [month],
            'Year': [year],
            'Total_Volume_Mn': [total_volume]
        })

        # Encode categorical variables
        input_data['UPI_Banks'] = label_encoder.transform(input_data['UPI_Banks'])

        # Perform the prediction using the loaded model
        prediction = lasso_model.predict(input_data)
        prediction_text = f"The predicted total value is: {prediction[0]:.2f}"
        
        return render_template('index.html', upi_banks=upi_banks, prediction_text=prediction_text)
    
    return render_template('index.html', upi_banks=upi_banks)

if __name__ == '__main__':
    app.run(debug=True)
