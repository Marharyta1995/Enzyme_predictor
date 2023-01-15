from fastapi import FastAPI, Depends
import torch

from model.cnn_model import initilize_model
from app.preprocessing_and_predict import predict
from app.pydantic_models import AA_Sequence
import numpy as np

# Initialize API Server
app = FastAPI(
    title="Enzyme Predictor",
    description="Enzyme prediction using AA sequence",
    version="0.0.1",
)


async def load_model():
    # Initialize the pytorch model
    cnn_model, optimizer = initilize_model(vocab_size=26,
                                           embed_dim=50,
                                           learning_rate=0.2,
                                           dropout=0.5)

    cnn_model.load_state_dict(torch.load("model/model.pt"))
    cnn_model.eval()

    return cnn_model


@app.post('/predict', tags=["Single sequence"], description="Make your prediction based on amino acid sequence")
async def do_predict(body: AA_Sequence, model = Depends(load_model)):

    # run model inference
    probs = predict(body.sequence, model=model)
    enzyme_result = np.argmax(probs)

    is_enzyme = {0: "not_enzyme", 1: "enzyme"}

    # prepare json for returning
    results = {
        'Sequence': body.sequence,
        'Class': is_enzyme[enzyme_result],
        'Probability_enzyme': f'{probs[1] * 100:.2f}',
    }

    return results
