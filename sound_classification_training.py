import argparse
import os
import torch
import torch.nn as nn
import audio_utils.audio_metadata as audio_metadata
from sound_classification_model import AudioClassifier, device
from sound_classification_inference import inference


# ----------------------------
# Training Loop
# ----------------------------
def training(model, train_dl, num_epochs):
  # Loss Function, Optimizer and Scheduler
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
  scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                steps_per_epoch=int(len(train_dl)),
                                                epochs=num_epochs,
                                                anneal_strategy='linear')

  # Make sure BatchNorm is in training mode (the model may have been in eval mode)
  model.train()

  # Repeat for each epoch
  for epoch in range(num_epochs):
    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0

    # Repeat for each batch in the training set
    for i, data in enumerate(train_dl):
        # Get the input features and target labels, and put them on the GPU
        inputs, labels = data[0].to(device), data[1].to(device)

        # Normalize the inputs
        inputs_m, inputs_s = inputs.mean(), inputs.std()
        inputs = (inputs - inputs_m) / inputs_s

        # Zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Keep stats for Loss and Accuracy
        running_loss += loss.item()

        # Get the predicted class with the highest score
        _, prediction = torch.max(outputs,1)
        # Count of predictions that matched the target label
        correct_prediction += (prediction == labels).sum().item()
        total_prediction += prediction.shape[0]

        if i % 10 == 0:    # print every 10 mini-batches (average loss so far this epoch)
           print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / (i + 1)))

    # Print stats at the end of the epoch
    num_batches = len(train_dl)
    avg_loss = running_loss / num_batches
    acc = correct_prediction/total_prediction
    print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')

  print('Finished Training')
  return epoch, avg_loss, acc


def main():
  parser = argparse.ArgumentParser(description="Train AudioClassifier on an 80/20 seeded split of UrbanSound8K")
  parser.add_argument("--epochs", type=int, default=1, help="number of epochs (default 1, just for demo)")
  parser.add_argument("--max-samples", type=int, default=None, help="train on a random subset (smoke tests)")
  parser.add_argument("--save-dir", default=str(audio_metadata.MODELS_DIR), help="where to save the .pth files")
  args = parser.parse_args()

  from sound_classification_split import get_dataloaders
  train_dl, val_dl = get_dataloaders(max_samples=args.max_samples)

  print(f"Using device: {device}")
  myModel = AudioClassifier().to(device)
  num_epochs = args.epochs
  epoch, avg_loss, acc = training(myModel, train_dl, num_epochs)

  save_dir = args.save_dir
  os.makedirs(save_dir, exist_ok=True)
  print("Saving model to:", save_dir)

  # Save weights after training
  torch.save(myModel.state_dict(), os.path.join(save_dir, f'mymodel_weights_epoch_{num_epochs}.pth'))

  # Save the entire model
  torch.save(myModel, os.path.join(save_dir, f'mymodel_entire_epoch_{num_epochs}.pth'))

  # Run inference on trained model with the validation set
  inference(myModel, val_dl)

  print(f'Training = Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')


# Only train when run as a script, not when another module imports training()
if __name__ == "__main__":
  main()
