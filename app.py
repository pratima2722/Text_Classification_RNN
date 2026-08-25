import io
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# ------------------------------------------------------------------
# Model Loading Logic
# ------------------------------------------------------------------
MODEL_PATH = "RNN.pkl"

def load_rnn_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_rnn_model()

# ------------------------------------------------------------------
# Single-File Frontend Layout (HTML + CSS Animations + JavaScript)
# ------------------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RNN Model Interface</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            min-height: 100vh;
            background: linear-gradient(-45deg, #0f172a, #1e1b4b, #311042, #0f172a);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #f8fafc;
            padding: 20px;
        }

        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .container {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            padding: 40px;
            width: 100%;
            max-width: 550px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
            animation: fadeIn 0.8s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        h1 {
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 8px;
            background: linear-gradient(90deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        p.subtitle {
            font-size: 0.9rem;
            color: #94a3b8;
            margin-bottom: 28px;
        }

        label {
            font-size: 0.85rem;
            font-weight: 600;
            color: #cbd5e1;
            display: block;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        textarea {
            width: 100%;
            height: 120px;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 12px;
            padding: 14px;
            color: #f8fafc;
            font-size: 0.95rem;
            outline: none;
            resize: none;
            transition: all 0.3s ease;
        }

        textarea:focus {
            border-color: #818cf8;
            box-shadow: 0 0 12px rgba(129, 140, 248, 0.3);
        }

        button {
            width: 100%;
            margin-top: 20px;
            padding: 14px;
            border: none;
            border-radius: 12px;
            background: linear-gradient(90deg, #6366f1, #4f46e5);
            color: #ffffff;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6);
        }

        button:active {
            transform: translateY(0);
        }

        .result-box {
            margin-top: 24px;
            padding: 16px;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            display: none;
            animation: slideUp 0.4s ease-out forwards;
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .result-title {
            font-size: 0.8rem;
            color: #94a3b8;
            text-transform: uppercase;
        }

        .result-value {
            font-size: 1.4rem;
            font-weight: 700;
            color: #38bdf8;
            margin-top: 4px;
        }

        .spinner {
            display: none;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 0.8s linear infinite;
            margin: 0 auto;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>RNN Sentiment Analysis</h1>
        <p class="subtitle">Deploy trained Keras Sequential SimpleRNN model on Vercel</p>
        
        <form id="predictionForm">
            <label for="inputData">Input Array / Sequence Data</label>
            <textarea id="inputData" placeholder="Enter comma-separated numbers (e.g., 1, 25, 43, 12...) matching input sequence length of 50"></textarea>
            
            <button type="submit" id="submitBtn">
                <span id="btnText">Run Inference</span>
                <div class="spinner" id="btnSpinner"></div>
            </button>
        </form>

        <div class="result-box" id="resultBox">
            <div class="result-title">Model Score / Probability</div>
            <div class="result-value" id="resultValue">--</div>
        </div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const btnText = document.getElementById('btnText');
            const btnSpinner = document.getElementById('btnSpinner');
            const resultBox = document.getElementById('resultBox');
            const resultValue = document.getElementById('resultValue');
            const inputVal = document.getElementById('inputData').value;

            // Parse comma-separated numbers into a numerical array
            const sequence = inputVal.split(',').map(num => parseFloat(num.trim())).filter(num => !isNaN(num));

            btnText.style.display = 'none';
            btnSpinner.style.display = 'block';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ sequence: sequence })
                });

                const data = await response.json();
                
                if (response.ok) {
                    resultValue.textContent = typeof data.prediction === 'number' 
                        ? data.prediction.toFixed(4) 
                        : JSON.stringify(data.prediction);
                } else {
                    resultValue.textContent = 'Error: ' + (data.error || 'Prediction failed');
                }
                resultBox.style.display = 'block';
            } catch (err) {
                resultValue.textContent = 'Server Connection Error';
                resultBox.style.display = 'block';
            } finally {
                btnText.style.display = 'inline';
                btnSpinner.style.display = 'none';
            }
        });
    </script>
</body>
</html>
"""

# ------------------------------------------------------------------
# Flask Application Routes
# ------------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded properly"}), 500
    
    try:
        data = request.get_json()
        raw_sequence = data.get("sequence", [])
        
        # RNN Input Layer sequence padding/truncating logic (50 timesteps)
        target_len = 50
        if len(raw_sequence) < target_len:
            padded_sequence = [0] * (target_len - len(raw_sequence)) + raw_sequence
        else:
            padded_sequence = raw_sequence[:target_len]
            
        input_array = np.array([padded_sequence], dtype=np.float32)
        
        prediction = model.predict(input_array)
        output_val = float(prediction[0][0])
        
        return jsonify({"prediction": output_val})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
