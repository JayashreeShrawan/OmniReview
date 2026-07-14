# TextCNN Training & Evaluation


#  Initialize TextCNN
cnn_model = TextCNN(embed_dim=EMBEDDING_DIM, num_classes=5).to(DEVICE)
print(cnn_model)

#  Model Parameter Summary
total_params = sum(p.numel() for p in cnn_model.parameters())
trainable_params = sum(p.numel() for p in cnn_model.parameters() if p.requires_grad)
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model size:           {total_params * 4 / 1e6:.2f} MB (float32)")

#  Loss Function and Optimizer
# CrossEntropy for multi-class classification
# L2 Regularization via weight_decay
cnn_criterion = nn.CrossEntropyLoss()
cnn_optimizer = optim.AdamW(cnn_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
cnn_scheduler = optim.lr_scheduler.ReduceLROnPlateau(cnn_optimizer, mode='min', factor=0.5, patience=3)
print(" Loss, optimizer, and LR scheduler configured")

#  CNN Training Function
def train_cnn(model, loader, optimizer, criterion):
    # Train the CNN for one epoch.
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in loader:
        x = batch['body_emb'].to(DEVICE)
        y = batch['rating'].to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total

# CNN Evaluation Function
def eval_cnn(model, loader, criterion):
    # Evaluate the CNN on a dataset.
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch['body_emb'].to(DEVICE)
            y = batch['rating'].to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    metrics = {
        'loss': total_loss / total,
        'accuracy': correct / total,
        'f1_macro': f1_score(all_labels, all_preds, average='macro')
    }
    return metrics, all_preds, all_labels

#  Execute CNN Training
cnn_history = TrainingHistory("TextCNN")
best_cnn_acc = 0

print(" Starting TextCNN Training...")
for epoch in range(1, NUM_EPOCHS_CNN + 1):
    with Timer(f"Epoch {epoch}"):
        train_loss, train_acc = train_cnn(cnn_model, train_emb_loader, cnn_optimizer, cnn_criterion)
        val_metrics, _, _ = eval_cnn(cnn_model, val_emb_loader, cnn_criterion)

        val_loss = val_metrics['loss']
        val_acc = val_metrics['accuracy']
        cnn_scheduler.step(val_loss)

        cnn_history.log(train_loss=train_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc)
        print(f"  Epoch {epoch}/{NUM_EPOCHS_CNN} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_cnn_acc:
            best_cnn_acc = val_acc
            save_checkpoint(cnn_model, cnn_optimizer, epoch, val_loss, CHECKPOINT_DIR / 'textcnn_best.pt')

#  Plot CNN Training History -- Loss
cnn_history.plot(['train_loss', 'val_loss'])

#  Plot CNN Training History -- Accuracy
cnn_history.plot(['train_acc', 'val_acc'])

# Load Best CNN Model and Evaluate on Test Set
load_checkpoint(cnn_model, cnn_optimizer, CHECKPOINT_DIR / 'textcnn_best.pt')
test_metrics, cnn_preds, cnn_labels = eval_cnn(cnn_model, test_emb_loader, cnn_criterion)

print(f"\n Test Results:")
print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
print(f"  Macro F1:  {test_metrics['f1_macro']:.4f}")
print(f"  Loss:      {test_metrics['loss']:.4f}")

results_collector.add_result('TextCNN_Classifier', test_metrics)

#  Classification Report
print("=== Detailed Classification Report ===")
target_names = ['1 Star', '2 Stars', '3 Stars', '4 Stars', '5 Stars']
print(classification_report(cnn_labels, cnn_preds, target_names=target_names))

#  Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Raw counts
cm = confusion_matrix(cnn_labels, cnn_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[1,2,3,4,5],
            yticklabels=[1,2,3,4,5], ax=axes[0])
axes[0].set_title('Confusion Matrix (Counts)', fontsize=14)
axes[0].set_xlabel('Predicted Rating')
axes[0].set_ylabel('Actual Rating')

# Normalized
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Oranges', xticklabels=[1,2,3,4,5],
            yticklabels=[1,2,3,4,5], ax=axes[1])
axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14)
axes[1].set_xlabel('Predicted Rating')
axes[1].set_ylabel('Actual Rating')

plt.tight_layout()
plt.show()

#  Per-Class Metrics Bar Chart
per_class = classification_report(cnn_labels, cnn_preds, target_names=target_names, output_dict=True)
per_class_df = pd.DataFrame({name: per_class[name] for name in target_names}).T[['precision', 'recall', 'f1-score']]

per_class_df.plot(kind='bar', figsize=(12, 6), colormap='Set2', edgecolor='black')
plt.title('Per-Class Precision, Recall, F1 -- TextCNN', fontsize=14)
plt.xlabel('Star Rating')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()

#CNN Summary
best_epoch, best_val = cnn_history.best('val_acc', mode='max')
print("=" * 60)
print(" Part 6 -- TextCNN Summary")
print("=" * 60)
print(f"  Architecture: 3-branch Conv1d (filters: 2,3,4) -> MaxPool -> FC")
print(f"  Parameters:   {trainable_params:,}")
print(f"  Best Epoch:   {best_epoch} (Val Acc: {best_val:.4f})")
print(f"  Test Acc:     {test_metrics['accuracy']:.4f}")
print(f"  Test F1:      {test_metrics['f1_macro']:.4f}")
print(" Part 6 Complete -- CNN Classifier trained and evaluated!")

