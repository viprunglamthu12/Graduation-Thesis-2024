import torch
from tqdm.auto import tqdm


def predict(model, dataset, batch_size=16):
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    y_hat = []

    with tqdm(total=len(dataset)) as pbar:
        for i in range(0, len(dataset), batch_size):
            batch = []
            for j in range(i, i+batch_size):
                if j < len(dataset):
                    batch.append(dataset[j])
                else:
                    break

            batch_text = [item[0] for item in batch]

            # Move batch tensor to GPU
            batch_tensor = torch.stack(batch_text, dim=0).to(device)

            # Perform computations on GPU
            outs = model(batch_tensor)
            for out in outs:
                if out[0].item() > out[1].item():
                    y_hat.append(0)
                else:
                    y_hat.append(1)
            
            pbar.update(len(batch))
            del batch, batch_text, batch_tensor, out

    return torch.tensor(y_hat)
