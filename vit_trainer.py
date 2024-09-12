import json, os, torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class VitTrainer:
    """
    Trainer class with progress bar, checkpointing, and runs logging.
    """

    def __init__(self, model, optimiser, loss_fn, exp_name, device, scheduler=None, base_dir="runs",
                 use_tensorboard=True):
        self.model = model.to(device)
        self.optimiser = optimiser
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.exp_name = exp_name
        self.device = device
        self.base_dir = base_dir

        # Create directories for checkpoints and logs
        self.runs_dir = os.path.join(base_dir, exp_name)
        os.makedirs(self.runs_dir, exist_ok=True)

        # TensorBoard writer for logging
        self.use_tensorboard = use_tensorboard
        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir=os.path.join(self.runs_dir, 'logs'))

    def save_checkpoint(self, epoch, is_best=False):
        """Save model, optimiser, and scheduler state."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimiser_state_dict': self.optimiser.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        }
        filename = f"model_best.pth" if is_best else f"model_{epoch}.pth"
        torch.save(checkpoint, os.path.join(self.runs_dir, filename))

    def load_checkpoint(self, checkpoint_name="model_best.pth"):
        """Load model and optimiser states."""
        checkpoint_path = os.path.join(self.runs_dir, checkpoint_name)
        if os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint '{checkpoint_name}'")
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            print(f"No checkpoint found at '{checkpoint_path}'")

    def save_run(self, config, train_losses, test_losses, accuracies):
        """Save run configuration and metrics."""
        # save config
        configfile = os.path.join(self.runs_dir, 'config.json')
        with open(configfile, 'w') as f:
            json.dump(config, f, sort_keys=True, indent=4)

        # save metrics
        jsonfile = os.path.join(self.runs_dir, 'metrics.json')
        with open(jsonfile, 'w') as f:
            data = {
                'train_losses': train_losses,
                'test_losses': test_losses,
                'accuracies': accuracies,
            }
            json.dump(data, f, sort_keys=True, indent=4)

    def train(self, trainloader, testloader, epochs, config, save_model_every_n_epochs=0, early_stopping_patience=20):
        train_losses, test_losses, accuracies = [], [], []
        best_loss = float('inf')
        patience = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            train_loss = self.train_epoch(trainloader)
            accuracy, test_loss = self.evaluate(testloader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)

            # step the scheduler
            if self.scheduler:
                self.scheduler.step(test_loss)

            # log metrics to TensorBoard
            if self.use_tensorboard:
                self.writer.add_scalar('Loss/Train', train_loss, epoch)
                self.writer.add_scalar('Loss/Test', test_loss, epoch)
                self.writer.add_scalar('Accuracy/Test', accuracy, epoch)

            print(f"Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")

            # early stopping and checkpointing
            if test_loss < best_loss:
                best_loss = test_loss
                patience = 0
                self.save_checkpoint(epoch + 1, is_best=True)
            else:
                patience += 1

            if patience >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            if save_model_every_n_epochs > 0 and (epoch + 1) % save_model_every_n_epochs == 0:
                print(f"\tSaving checkpoint at epoch {epoch + 1}")
                self.save_checkpoint(epoch + 1)

        # save final model
        self.save_checkpoint(epochs, is_best=False)
        # save run details
        self.save_run(config, train_losses, test_losses, accuracies)

    def train_epoch(self, trainloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(trainloader, desc="Training", leave=False)

        for batch in progress_bar:
            batch = [t.to(self.device) for t in batch]
            images, labels = batch

            self.optimiser.zero_grad()
            loss = self.loss_fn(self.model(images)[0], labels)
            loss.backward()
            self.optimiser.step()

            total_loss += loss.item() * len(images)
            progress_bar.set_postfix(loss=loss.item())

        return total_loss / len(trainloader.dataset)

    @torch.no_grad()
    def evaluate(self, testloader):
        """Evaluate on test data."""
        self.model.eval()
        total_loss = 0
        correct = 0
        progress_bar = tqdm(testloader, desc="Evaluating", leave=False)

        for batch in progress_bar:
            batch = [t.to(self.device) for t in batch]
            images, labels = batch

            logits, _ = self.model(images)
            loss = self.loss_fn(logits, labels)
            total_loss += loss.item() * len(images)

            predictions = torch.argmax(logits, dim=1)
            correct += torch.sum(predictions == labels).item()

            progress_bar.set_postfix(loss=loss.item())

        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)
        return accuracy, avg_loss
