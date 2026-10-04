# Protocol terminology clarification

The inherited `scope` string says “nonlinear challenger”. The actual frozen
`semantic_model` and source define a **linear logistic classifier after
training-only scaling/PCA**, augmented with the five scalar features. No
nonlinear classifier or encoder fine-tuning is used in this semantic study.
This clarification corrects terminology without changing frozen inputs,
implementation, coefficients, thresholds or quality gates. It was written
before head fitting or viewing semantic-head scores. Prior nonlinear tree
experiments remain separately identified and unchanged.
