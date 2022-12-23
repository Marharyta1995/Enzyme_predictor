#Makes a predict based on the aa sequence"
import torch
import torch.nn.functional as F

def predict(seq, model, max_len = 500):
    """Predict probability that a protein sequence is an enzyme."""
    model.to("cpu")
    # Tokenize, pad and encode text
    aa_code = 'ABCDEFGHIKLMNOPQRSTUVWYXZ'
    char_to_int = dict((c, i) for i, c in enumerate(aa_code, start=1))
    encoded_seq = [char_to_int[char] for char in seq]

    padding = 0
    if len(seq) < max_len:
        for i in range(max_len + 1 - len(seq)):
            encoded_seq.append(padding)

    # Convert to PyTorch tensors
    input_id = torch.tensor(encoded_seq).unsqueeze(dim=0)

    # Compute logits
    logits = model.forward(input_id)

    #  Compute probability
    probs = F.softmax(logits, dim=1).squeeze(dim=0)
    probs_detached = probs.detach().numpy()


    return probs_detached