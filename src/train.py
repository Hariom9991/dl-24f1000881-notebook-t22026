# training for scratch model

def train_scratch_model(model, train_loader, val_loader, val_main_df, epochs=5, lr=2e-3, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    history = []
    best_map3 = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        for batch in tqdm(train_loader, desc=f'Scratch Epoch {epoch+1}'):
            input_ids = batch['input_ids'].to(device)
            targets = batch['target'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            total_train_loss += loss.item()
        val_metrics = evaluate_scratch_model(model, val_loader, val_main_df, device)
        
        epoch_result = {
            'epoch': epoch + 1,
            'train_loss': total_train_loss / len(train_loader),
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'val_f1_macro': val_metrics['f1'],
            'val_map3': val_metrics['map3']
        }
        history.append(epoch_result)
        
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": epoch_result["train_loss"],
            "val/loss": epoch_result["val_loss"],
            "val/accuracy": epoch_result["val_accuracy"],
            "val/f1_macro": epoch_result["val_f1_macro"],
            "val/map3": epoch_result["val_map3"]
        })

        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {epoch_result['train_loss']:.5f}")
        print(f"Val Loss: {epoch_result['val_loss']:.5f} | Val Acc: {epoch_result['val_accuracy']:.5f} | Val F1 Macro: {epoch_result['val_f1_macro']:.5f} | Val MAP@3: {epoch_result['val_map3']:.5f}")

        if val_metrics['map3'] > best_map3:
            best_map3 = val_metrics['map3']
            best_state = copy.deepcopy(model.state_dict())  # deepcopy: state_dict() alone returns live tensor
                                                              # references that keep changing as training continues

    if best_state is not None:
        model.load_state_dict(best_state)
        
    return model, pd.DataFrame(history)


# training code for pretrained model

def train_electra_mc(
    model,
    train_loader,
    val_loader,
    epochs=4,
    lr=2e-5,
    weight_decay=0.01,
    device="cpu",
    class_weights=None,
    accumulation_steps=1
):
    model = model.to(device)
    weights = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    history = []
    best_map3 = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        train_logits = []
        train_labels = []
        optimizer.zero_grad()

        progress_bar = tqdm(train_loader, desc=f"ELECTRA Epoch {epoch+1}")

        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }

            if "token_type_ids" in batch:
                model_inputs["token_type_ids"] = batch["token_type_ids"].to(device)

            outputs = model(**model_inputs)
            logits = outputs.logits
            loss = criterion(logits, labels)

            (loss / accumulation_steps).backward()

            is_last_batch = (step + 1) == len(train_loader)
            if (step + 1) % accumulation_steps == 0 or is_last_batch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_train_loss += loss.item()
            train_logits.append(logits.detach().cpu().numpy())
            train_labels.append(labels.detach().cpu().numpy())

        train_logits = np.concatenate(train_logits, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        train_metrics, _, _ = compute_multiclass_metrics(train_labels, train_logits)

        val_metrics = evaluate_electra_mc(model, val_loader, device, class_weights=class_weights)

        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": total_train_loss / len(train_loader),
            "train_accuracy": train_metrics["accuracy"],
            "train_f1": train_metrics["f1_macro"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1"], 
            "val_map3": val_metrics["map3"]
        }
        history.append(epoch_result)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {epoch_result['train_loss']:.5f} | Train Acc: {epoch_result['train_accuracy']:.5f} | Train F1: {epoch_result['train_f1']:.5f}")
        print(f"Val   Loss: {epoch_result['val_loss']:.5f} | Val   Acc: {epoch_result['val_accuracy']:.5f} | Val F1 Macro: {epoch_result['val_f1_macro']:.5f} | Val MAP@3: {epoch_result['val_map3']:.5f}")

        if epoch_result["val_map3"] > best_map3:
            best_map3 = epoch_result["val_map3"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    history_df = pd.DataFrame(history)
    return model, history_df