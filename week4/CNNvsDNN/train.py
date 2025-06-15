from tqdm import tqdm
import torch

torch.manual_seed(42)  # For reproducibility

def train(model, device, train_loader, test_loader, criterion, optimizer, epochs=10):

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        running_train_loss = 0.0

        # --- Training Step ---
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * data.size(0)

        avg_train_loss = running_train_loss / len(train_loader.dataset)

        # --- Evaluation Step ---

        model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)

        tqdm.write(f"Epoch {epoch+1}/{epochs}, "
                   f"Train Loss: {avg_train_loss:.4f}, "
                   f"Test Loss: {test_loss:.4f}, "
                     f"Accuracy: {accuracy:.2f}%")
