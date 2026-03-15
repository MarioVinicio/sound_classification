import os
import torch
from sound_classification_model import AudioClassifier
from sound_classification_inference import inference
from sound_classification_split import val_dl


# Load model later for inference
myModel = AudioClassifier()
save_dir = os.getcwd() + "/sound_classification/trained_models"
myModel.load_state_dict(torch.load(os.path.join(save_dir, 'mymodel_weights_epoch_30.pth'), weights_only=True))
myModel.eval()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f'device = {device}')
myModel.to(device)

inference(myModel, val_dl)