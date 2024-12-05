import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold

from core.TSTR_scores_da.utils import augment_batch, get_dataloader
from core.TSTR_scores_da.models import CORAL, TSTRClassifier




def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = loss_fn(outputs, y_batch)
            total_loss += loss.item()

            _, predicted_labels = torch.max(outputs, 1)
            all_preds.extend(predicted_labels.detach().cpu().numpy())
            all_labels.extend(y_batch.detach().cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    total_loss /= len(test_loader)

    return total_loss, accuracy, f1, cm



def train_model(model, train_loader, optimizer, num_epochs, augment=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            if augment:
                x_batch = augment_batch(x_batch)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss /= len(train_loader)

        if (epoch+1) % 10 == 0:
            print(f"\tEpoch {epoch + 1}/{num_epochs} - Train loss: {total_loss:.4f}")

    return model



def train_model_coral(model, train_loader, x_test, optimizer, num_epochs, coral_weight, augment=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    x_test = torch.tensor(x_test, dtype=torch.float32, device=device).to(device)

    loss_fn = nn.CrossEntropyLoss()
    loss_coral = CORAL()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_class_loss = 0
        total_coral_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            if augment:
                x_batch = augment_batch(x_batch)
            optimizer.zero_grad()
            outputs = model(x_batch)
            class_loss = loss_fn(outputs, y_batch)
            coral_loss = loss_coral(model.feature_extractor(x_batch), model.feature_extractor(x_test))
            loss = class_loss + coral_weight * coral_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_class_loss += class_loss.item()
            total_coral_loss += coral_loss.item()
        total_loss /= len(train_loader)
        total_class_loss /= len(train_loader)
        total_coral_loss /= len(train_loader)

        if (epoch+1) % 10 == 0:
            print(f"\tEpoch {epoch + 1}/{num_epochs} - Train loss: {total_loss:.4f} ({total_class_loss:.4f} + {total_coral_loss:.2e})")

    return model



def train_only(x_train, y_train, num_epochs, augment=False):
    assert len(np.unique(y_train)) == 4, f"Labels are not complete: {np.unique(y_train)}"

    num_classes = len(np.unique(y_train))

    train_loader = get_dataloader(x_train, y_train, shuffle=True)

    model = TSTRClassifier(num_timesteps=128, num_channels=3, num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    trained_model = train_model(model, train_loader, optimizer, num_epochs, augment)

    return trained_model



def train_and_test(x_train, y_train, x_test, y_test, num_epochs, augment=False, coral_weight=0):
    assert np.array_equal(np.unique(y_train), np.unique(y_test)), f"Training and test labels do not match: {np.unique(y_train)} vs {np.unique(y_test)}"
    assert len(np.unique(y_train)) == 4, f"Labels are not complete: {np.unique(y_train)}"

    num_classes = len(np.unique(y_train))

    train_loader = get_dataloader(x_train, y_train, shuffle=True)
    test_loader = get_dataloader(x_test, y_test)

    model = TSTRClassifier(num_timesteps=128, num_channels=3, num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    if coral_weight == 0:
        trained_model = train_model(model, train_loader, optimizer, num_epochs, augment)
    else:
        trained_model = train_model_coral(model, train_loader, x_test, optimizer, num_epochs, coral_weight, augment)

    test_loss, test_accuracy, test_f1, test_cm = evaluate_model(trained_model, test_loader)

    return test_loss, test_accuracy, test_f1, test_cm

    

def train_cv(x_train, y_train, num_epochs):
    assert len(np.unique(y_train)) == 4, f"Labels are not complete: {np.unique(y_train)}"
    
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2710)

    losses = []
    accs = []
    f1s = []
    total_cm = None

    for train_index, test_index in skf.split(x_train, y_train):
        x_train_fold, x_test_fold = x_train[train_index], x_train[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

        loss, acc, f1, cm = train_and_test(x_train_fold, y_train_fold, x_test_fold, y_test_fold, num_epochs)
        losses.append(loss)
        accs.append(acc)
        f1s.append(f1)
        if total_cm is None:
            total_cm = cm
        else:
            total_cm += cm

    return np.mean(losses), np.mean(accs), np.mean(f1s), total_cm / n_splits



def pseudo_labeling(model, x, y, num_epochs, augment=False):
    raise ValueError
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_pl, x_test, y_pl, y_test = train_test_split(x, remap_labels(y), test_size=0.2, stratify=y, shuffle=True, random_state=seed)
    print(f'x_pl.shape: {x_pl.shape} | np.unique(y_pl): {np.unique(y_pl)}')
    print(f'x_test.shape: {x_test.shape} | np.unique(y_test): {np.unique(y_test)}')

    x_pl = torch.tensor(x_pl, dtype=torch.float32, device=device)
    y_pl_pred = torch.argmax(model(x_pl), dim=1).detach().cpu().numpy()
    x_pl = x_pl.detach().cpu().numpy()

    return train_and_test(x_pl, y_pl_pred, x_test, y_test, dataset, num_epochs=num_epochs, augment=augment, patience=patience)



def fine_tune(model, x_train, y_train, num_epochs):
    # Freeze feature extraction layers
    for name, param in model.named_parameters():
        if 'conv' in name or 'bn' in name:
            param.requires_grad = False

    train_loader = get_dataloader(x_train, y_train, shuffle=True)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

    trained_model = train_model(model, train_loader, optimizer, num_epochs)

    return trained_model