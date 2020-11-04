def run_evaluations():
    # evaluate precision, recall with confidence scores
    confidence_ordering = l2_confidence_scores.argsort()[::-1]  
    scores = (preds == targets).all(axis=1)
    precisions = evaluate_lr_precision(scores[confidence_ordering])

    # compute prediction accuracy
    overall_acc, l1_overall_acc, l1_class_accs = \
        evaluate_batch_predictions(preds, targets, len(taxonomy._root.children))

    # log results
    logger.debug("Overall Accuracy:", overall_acc)
    logger.debug("L1 Accuracy:", l1_overall_acc)

    logger.debug("Accuracies by Category:")
    for class_id, l1_class_acc in l1_class_accs.items():
        class_name = taxonomy.class_ids_to_category_name([class_id])
        logger.debug(f"L1 Category {class_name}:", l1_class_acc)

    # aggregate results
    results = pd.concat([pd.DataFrame({
        'Scores': scores,
        'Left precision': precisions[:, 0], 
        'Right precision': precisions[:, 1], 
        'Confidence': l2_confidence_scores, 
        'L1 Confidence': l1_confidence_scores,
        'L1_target': test_data.data["L1"],
        'L2_target': test_data.data["L2"],
    }), preds], axis=1)
    results.to_csv(join(args.save_dir, "results.csv"), index=False)
