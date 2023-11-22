# Transfer Learning for End-to-End Revenue Maximization

## Introduction
In the landscape of machine learning and predictive analytics, decision-makers are often constrained by the limited quantity of labeled "complete" data. This project addresses the challenge of predicting local store demand and maximizing revenue with incomplete feature spaces, as commonly seen in e-commerce to offline retail transitions. A novel hybrid transfer learning algorithm is proposed, aiming to learn transferable features and provide a sub-optimal pricing policy with objective optimality.

## Files Description
- `code/`: Directory containing all the source code files.
- `Data_preprocessing.ipynb`: Jupyter notebook for preprocessing the data used in the models.
- `baseline.py`: Python script implementing the first estimate then optimize framework using linear regression and Multi-Layer Perceptron (MLP) models.
- `draw_graph.py`: Python script to visualize results and draw graphs for analysis.
- `transfer.py`: Python script for implementing our transfer learning for end-to-end training using monotone neural networks.

## Documentation
- `E2E pricing.pdf`: A document providing a brief summary of the problem and the model used for end-to-end pricing strategy.

## Dataset
- `JDdata.zip`: Compressed file containing the real-world data utilized in our experiments.

## How to Use
1. Consult `E2E pricing.pdf` for an in-depth understanding of the problem and the modeling approach.
2. Begin by preprocessing your dataset using the `Data_preprocessing.ipynb` notebook to fit the expected input format.
3. Run `baseline.py` to establish a baseline model for your prediction task.
4. Apply `transfer.py` to perform the transfer learning-based end-to-end training with the proposed monotone neural network.
5. Utilize `draw_graph.py` to generate visual representations of your data and results.

## Contribution
We encourage contributions to this project. If you have suggestions or improvements, please fork the repository and submit a pull request.

## Citation
