from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import io

# Load your trained model
rfc_model = joblib.load("E:/Kiffya_10_acc/Week 4/rossmann-sales-prediction/model/random_forest_model-23-09-2024-19-28-47.pkl")

app = FastAPI()

# Define the required columns for prediction
required_columns = [
    'Id', 'Store', 'DayOfWeek', 'Open', 'Promo', 'SchoolHoliday',
    'CompetitionDistance', 'CompetitionOpenSinceMonth',
    'CompetitionOpenSinceYear', 'Promo2', 'Weekday', 'IsWeekend', 'Day',
    'Month', 'Year', 'IsHoliday', 'StoreType_b', 'StoreType_c',
    'StoreType_d', 'Assortment_b', 'Assortment_c'
]

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the file contents and load it into a DataFrame
        contents = await file.read()
        test_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # Check if all required columns exist in the uploaded CSV
        missing_columns = [col for col in required_columns if col not in test_df.columns]
        
        if missing_columns:
            return JSONResponse(status_code=400, content={"error": f"Missing columns: {', '.join(missing_columns)}"})

        # Ensure 'Id' column is present and correct
        if 'Id' not in test_df.columns:
            test_df['Id'] = range(1, len(test_df) + 1)

        # Make predictions (assumes 'Id' is not part of the features used for prediction)
        predictions = rfc_model.predict(test_df.drop(columns='Id'))

        # Create a results DataFrame with 'Id' and predictions
        # results_df = pd.DataFrame({
        #     'Id': test_df['Id'],
        #     'Prediction': predictions
        # })

        # Return the results as JSON response
        return JSONResponse(content={"Predictions": predictions.tolist()})

    except pd.errors.ParserError:
        return JSONResponse(status_code=400, content={"error": "Failed to parse the CSV file."})
    except ValueError as ve:
        return JSONResponse(status_code=400, content={"error": f"Value error: {str(ve)}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"An error occurred: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
