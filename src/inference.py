#inference for the bes model
mc_model.eval()
ranked_predictions = []

with torch.no_grad():
    for batch in tqdm(test_loader_mc, desc="ELECTRA Test Inference"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        if "token_type_ids" in batch:
            model_inputs["token_type_ids"] = batch["token_type_ids"].to(DEVICE)

        outputs = mc_model(**model_inputs)
        logits = outputs.logits.detach().cpu().numpy()
        probs = stable_softmax(logits)

        for row_probs in probs:
            top3_idx = np.argsort(-row_probs)[:3]
            ranked_predictions.append(" ".join([ID2LABEL[i] for i in top3_idx]))

submission = pd.DataFrame({
    "ID": test_df["id"].values,
    "Prediction": ranked_predictions
})

display(submission.head())
submission.to_csv("submission_electra_mc.csv", index=False)
print("submission_electra_mc.csv saved successfully")