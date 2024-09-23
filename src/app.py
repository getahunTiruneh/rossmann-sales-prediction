from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import pandas as pd
import joblib
import io

# Load your trained model
rfc_model = joblib.load("../notebook/random_forest_model-23-09-2024-19-28-47.pkl")

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def main():
    content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Upload CSV for Prediction</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 50px;
            }
            h1 {
                color: #333;
            }
        </style>
    </head>
    <body>
        <h1>Upload CSV for Prediction</h1>
        <form id="upload-form" enctype="multipart/form-data">
            <input name="file" type="file" accept=".csv" required>
            <button type="submit">Submit</button>
        </form>
        <div id="result"></div>

        <script>
            document.getElementById("upload-form").onsubmit = async function(event) {
                event.preventDefault();
                const formData = new FormData(this);
                const response = await fetch("/predict/", {
                    method: "POST",
                    body: formData
                });
                const result = await response.json();
                document.getElementById("result").innerText = JSON.stringify(result, null, 2);
            };
        </script>
    </body>
    </html>
    """
    return content

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the file contents and load it into a DataFrame
        contents = await file.read()
        test_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # Ensure the DataFrame has an 'Id' column; if not, add it
        if 'Id' not in test_df.columns:
            test_df['Id'] = range(1, len(test_df) + 1)

        # Make predictions (assumes 'Id' is not part of features)
        predictions = rfc_model.predict(test_df.drop(columns='Id'))

        # Create a results DataFrame with Id and predictions
        results_df = pd.DataFrame({
            'Id': test_df['Id'],
            'Prediction': predictions
        })

        # Return the results as JSON
        return results_df.to_dict(orient='records')

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
