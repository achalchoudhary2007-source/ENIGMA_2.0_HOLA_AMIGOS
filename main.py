import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# Allow the frontend to communicate with this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrainRequest(BaseModel):
    learning_rate: float
    epochs: int
    degree: int = 1
    X: list
    y: list

def poly_features(X, degree):
    """Helper function to create polynomial features for regression."""
    return np.column_stack([X**i for i in range(1, degree + 1)])

@app.get("/generate-data")
def generate_data(type: str = "regression"):
    """Generates random datasets based on the selected algorithm."""
    try:
        np.random.seed(42)
        if type == "regression":
            # 1D array for polynomial curve fitting
            X = np.sort(np.random.uniform(-1, 1, 50))
            y = 2 * (X**2) - 0.5 * X + np.random.normal(0, 0.2, 50)
        else:
            # 2D array for binary classification (two clusters)
            X = np.random.uniform(-1, 1, (50, 2))
            y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        return {"X": X.tolist(), "y": y.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train-linear")
def train_linear(req: TrainRequest):
    """Executes Gradient Descent for Polynomial Regression."""
    try:
        X_np = np.array(req.X)
        y_np = np.array(req.y).reshape(-1, 1)
        
        X_poly = poly_features(X_np, req.degree)
        X_poly = np.c_[np.ones(X_poly.shape[0]), X_poly] # Add bias term (intercept)
        
        m, n = X_poly.shape
        W = np.random.randn(n, 1) * 0.1
        history = []
        
        for epoch in range(req.epochs):
            predictions = X_poly.dot(W)
            errors = predictions - y_np
            loss = (1 / (2 * m)) * np.sum(errors**2)
            
            gradients = (1 / m) * X_poly.T.dot(errors)
            W -= req.learning_rate * gradients
            
            # Save frames for the UI animation
            if epoch % max(1, req.epochs // 50) == 0 or epoch == req.epochs - 1:
                X_curve = np.linspace(-1.2, 1.2, 100)
                X_curve_poly = np.c_[np.ones(100), poly_features(X_curve, req.degree)]
                y_curve = X_curve_poly.dot(W).flatten()
                
                history.append({
                    "epoch": epoch,
                    "loss": float(loss),
                    "boundary": [{"x": float(xc), "y": float(yc)} for xc, yc in zip(X_curve, y_curve)]
                })
                
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Math Error: {str(e)}")

@app.post("/train-logistic")
def train_logistic(req: TrainRequest):
    """Executes Gradient Descent for Logistic Classification."""
    try:
        X_np = np.array(req.X)
        y_np = np.array(req.y).reshape(-1, 1)
        X_b = np.c_[np.ones(X_np.shape[0]), X_np] # Add bias term
        
        m, n = X_b.shape
        W = np.random.randn(n, 1) * 0.1
        history = []
        
        for epoch in range(req.epochs):
            z = X_b.dot(W)
            # Sigmoid function with clipping to prevent overflow
            predictions = 1 / (1 + np.exp(-np.clip(z, -250, 250)))
            
            # Binary Cross-Entropy Loss
            loss = -np.mean(y_np * np.log(predictions + 1e-15) + (1 - y_np) * np.log(1 - predictions + 1e-15))
            gradients = (1 / m) * X_b.T.dot(predictions - y_np)
            W -= req.learning_rate * gradients
            
            if epoch % max(1, req.epochs // 50) == 0 or epoch == req.epochs - 1:
                # Calculate coordinates for the decision boundary line
                x_bnd = np.array([-1.2, 1.2])
                # Added 1e-8 to prevent division by zero if W[2,0] is exactly 0
                y_bnd = -(W[0,0] + W[1,0]*x_bnd) / (W[2,0] + 1e-8)
                
                history.append({
                    "epoch": epoch, 
                    "loss": float(loss),
                    "boundary": [{"x": float(x_bnd[0]), "y": float(y_bnd[0])}, {"x": float(x_bnd[1]), "y": float(y_bnd[1])}]
                })
                
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Math Error: {str(e)}")

if __name__ == "__main__":
    # Ensure this is running on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)