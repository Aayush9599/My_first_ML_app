from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import os


# Load the saved model
model = joblib.load("gradient_boosting_pipeline.pkl")

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index_1.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        customers_per_week = int(request.form['customers_per_week'])
        promotion = int(request.form['promotion'])
        weather = request.form['weather']
        events = int(request.form['events'])
        holiday = int(request.form['holiday'])
        day_of_week = request.form['day_of_week']
        temperature = float(request.form['temperature'])
        restaurant_rating = float(request.form['restaurant_rating'])
        special_promotion = int(request.form['special_promotion'])

        # Prepare input data
        input_data = {
            'Customers per Week': [customers_per_week],
            'Promotion': [promotion],
            'Weather': [weather],
            'Event': [events],
            'Holiday': [holiday],
            'Day of Week': [day_of_week],
            'Temperature (C)': [temperature],
            'Restaurant Rating': [restaurant_rating],
            'Special Promotion': [special_promotion]
        }

        # Predict using the model
        prediction = model.predict(pd.DataFrame(input_data))
#------------------------------------------------------------------#
        # Format the prediction results
        result = {
            'Chicken Consumption (kg)': prediction[0][0],
            'Beef Consumption (kg)': prediction[0][1],
            'Fish Consumption (kg)': prediction[0][2],
            'Pork Consumption (kg)': prediction[0][3]
        }

        # Return the result
        # return render_template('result.html', prediction=result)

#------- Adding visuals -------------------------------------------#

        # Labels for different meats
        labels = ['Chicken', 'Beef', 'Fish', 'Pork']
        values = prediction[0].tolist()  # Extract the values
        
        # Generate a bar chart
        plt.figure(figsize=(8, 5))
        bars = plt.bar(labels, values, color=['orange', 'brown', 'blue', 'red'])

        # Add text annotations on top of the bars
        for i, value in enumerate(values):
            plt.text(i, value + 0.5, f'{value:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')


        plt.xlabel("Meat Type")
        plt.ylabel("Predicted Consumption (kg)")
        plt.title("Predicted Meat Consumption")
        plt.ylim(0, max(prediction[0]) + 10)

        # Save the chart in the static folder
        img_path = "static/prediction_chart.png"
        plt.savefig(img_path)
        plt.close()

        # return render_template('result.html', prediction_text=f'Predicted Consumption:', img_path=img_path)

        # Return the template with the result and image
        return render_template('result_1.html', prediction=result, img_path=img_path)

    except Exception as e:
        return render_template('index_1.html', prediction_text=f'Error: {str(e)}')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
