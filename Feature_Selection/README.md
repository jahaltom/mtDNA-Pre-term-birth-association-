## Consensus Feature Table Generator

ConsensusFeatureTable.py: Aggregates feature-importance outputs from Random Forest, Gradient Boosting, Neural Network, Elastic Net, and Ridge Logistic models into a single consensus covariate-ranking table. The script reads GA and PTB SHAP-importance files, cleans transformed feature names, optionally collapses one-hot encoded categories back to their parent variable, and ranks variables across model types. It outputs ConsensusFeatureTable.csv and ConsensusFeatureTable.tsv, including per-model ranks/importances, number of models where each variable appears in the top 10 or top 20, and mean/median rank across models.

This table is intended for informative covariate screening, not prediction, and helps identify variables consistently supported across multiple algorithms and outcomes.
