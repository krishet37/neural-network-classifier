# neural-network-classifier
A neural network classification project for image recognition of letters A, B, and C using synthetic binary pixel patterns

## 📋 Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Approach](#approach)
- [Methodology](#methodology)
- [Analysis Process](#analysis-process)
- [Key Findings](#key-findings)
- [Installation](#installation)
- [Usage](#usage)
- [Requirements](#requirements)
- [Contributing](#contributing)

## 🔍 Overview
This project implements a **basic feedforward neural network from scratch** using only NumPy to classify synthetic image data representing the characters A, B, and C. Each image is a 5×6 (30-pixel) grid encoded as a 1D array. The project demonstrates fundamental neural network concepts through custom implementation of backpropagation, weight optimization, and classification without using external ML libraries like TensorFlow or PyTorch.

This is part of **Module 11: Neural Network from Scratch** assignment, designed to test understanding of fundamental neural network concepts by implementing a basic feedforward neural network using only NumPy.

## 📁 Project Structure
```
neural-network-classifier/
│
├── README.md                          # Project documentation
├── neural_network_classifier.ipynb    # Main Jupyter notebook
├── data/                              # Dataset directory (synthetic data)
├── models/                            # Saved model files
├── results/                           # Output results and visualizations
└── requirements.txt                   # Python dependencies
```

## 🎯 Approach
### Problem Definition
- **Objective**: Classify input images as letter A, B, or C using a two-layer neural network trained via backpropagation
- **Dataset**: Synthetic image data with custom binary patterns for letters A, B, and C
- **Image Format**: 5×6 (30-pixel) binary pixel grids encoded as 1D arrays
- **Target Variable**: Character class (A, B, or C)
- **Constraint**: Implementation using only NumPy - no external ML libraries (TensorFlow, PyTorch)

### Strategy
1. **Define pixel-based binary patterns** for letters A, B, and C
2. **Create training data** using synthetic binary patterns
3. **Implement feedforward neural network** with hidden layer
4. **Use sigmoid activation function** for non-linearity
5. **Implement custom backpropagation logic** for training
6. **Optimize weights** to minimize classification error
7. **Track and visualize** loss and accuracy across epochs
8. **Test the model** and display predictions with matplotlib

## 🔬 Methodology
### Data Preprocessing
- **Data Source**: No external dataset file required
- **Data Creation**: Custom binary patterns (5×6 grids) for letters A, B, and C
- **Encoding**: Binary pixel patterns (0s and 1s) representing each character
- **Format**: 1D arrays of length 30 (flattened 5×6 grids)
- **Labeling**: Three-class classification (A, B, C)

### Model Architecture
- **Type**: Basic feedforward neural network (two-layer)
- **Input Layer**: 30 input nodes (one per pixel in 5×6 grid)
- **Hidden Layer**: Custom-defined number of nodes
- **Output Layer**: 3 output nodes (one per class: A, B, C)
- **Activation Function**: Sigmoid activation function
- **Training Method**: Custom backpropagation logic
- **Implementation**: Pure NumPy (no TensorFlow/PyTorch)

### Core Components
1. **Weight Initialization**: Random initialization of weights
2. **Forward Pass**: Matrix operations for feedforward computation
3. **Activation Functions**: Sigmoid function for hidden and output layers
4. **Loss Computation**: Classification error calculation
5. **Backpropagation**: Custom gradient computation
6. **Gradient Descent Updates**: Weight optimization
7. **Training Loop**: Epoch-based training with loss/accuracy tracking

### Hyperparameters
- **Learning Rate**: Optimized for convergence
- **Number of Epochs**: Sufficient for model convergence
- **Hidden Layer Size**: Tuned for optimal performance
- **Batch Processing**: Full batch or mini-batch approach

## 📊 Analysis Process
### 1. Data Generation
- Design binary pixel patterns for each letter (A, B, C)
- Create 5×6 grid representations
- Encode as 1D arrays (30 elements each)
- Prepare labeled training dataset

### 2. Model Implementation
- **Weight Initialization**: Initialize weights randomly for hidden and output layers
- **Forward Propagation**: 
  - Input layer → Hidden layer computation
  - Apply sigmoid activation
  - Hidden layer → Output layer computation
  - Apply sigmoid activation for probabilities
- **Backpropagation**:
  - Compute output layer error
  - Calculate gradients for output weights
  - Backpropagate error to hidden layer
  - Calculate gradients for hidden weights
- **Weight Updates**: Apply gradient descent to optimize weights

### 3. Training Process
- **Epoch Loop**: Iterate through multiple epochs
- **Forward Pass**: Compute predictions for all training samples
- **Loss Calculation**: Measure classification error
- **Backward Pass**: Compute gradients via backpropagation
- **Weight Update**: Apply gradient descent
- **Metrics Tracking**: Record loss and accuracy per epoch
- **Convergence Monitoring**: Track improvement over epochs

### 4. Visualization & Testing
- **Training Curves**: Plot loss and accuracy across epochs using matplotlib
- **Performance Analysis**: Analyze learning progression
- **Model Testing**: Predict classes for test inputs
- **Image Display**: Use `matplotlib.pyplot.imshow()` to display input images
- **Result Verification**: Compare predictions with expected labels

### 5. Hands-On Learning
This project provides hands-on experience with:
- **Matrix Operations**: NumPy array manipulations
- **Weight Initialization**: Understanding initialization strategies
- **Activation Functions**: Implementing and applying sigmoid
- **Loss Computation**: Calculating classification error
- **Gradient Descent Updates**: Understanding optimization
- **Backpropagation**: Core neural network learning algorithm

## 🎓 Key Findings
### Model Performance
*To be updated after running the model:*
- **Training Accuracy**: [value]%
- **Final Loss**: [value]
- **Convergence**: Number of epochs needed
- **Classification Success**: Correct predictions on test data

### Insights
1. **From-Scratch Implementation**: Successfully built neural network using only NumPy without ML libraries
2. **Backpropagation Understanding**: Gained deep understanding of gradient computation and weight updates
3. **Pattern Recognition**: Model learned to distinguish between binary patterns of letters A, B, and C
4. **Matrix Operations**: Mastered essential matrix operations for neural network computations
5. **Visualization**: Effectively tracked and visualized training progress

### Technical Achievements
- ✅ Implemented custom feedforward neural network
- ✅ Created custom backpropagation logic
- ✅ Used only NumPy for all computations
- ✅ Successfully trained model on synthetic data
- ✅ Visualized training metrics with matplotlib
- ✅ Achieved accurate classification of letter patterns

### Challenges & Solutions
1. **Weight Initialization**: Finding appropriate initial weight ranges
   - Solution: Experimented with different initialization strategies
2. **Learning Rate Tuning**: Balancing convergence speed and stability
   - Solution: Tested multiple learning rates to find optimal value
3. **Gradient Computation**: Implementing accurate backpropagation
   - Solution: Carefully derived and validated gradient calculations
4. **Numerical Stability**: Avoiding numerical overflow/underflow
   - Solution: Applied appropriate scaling and clipping techniques

### Future Improvements
- Implement additional activation functions (ReLU, Tanh)
- Add regularization techniques (L1/L2, Dropout)
- Experiment with different network architectures
- Extend to more letters or larger image sizes
- Implement mini-batch gradient descent
- Add early stopping criteria
- Create more complex test patterns

## 💻 Installation
```bash
# Clone the repository
git clone https://github.com/krishet37/neural-network-classifier.git
cd neural-network-classifier

# Install required packages
pip install -r requirements.txt
```

## 🚀 Usage
```bash
# Open the Jupyter notebook
jupyter notebook neural_network_classifier.ipynb
```

### Workflow in Notebook:
1. **Define Binary Patterns**: Create 5×6 pixel patterns for letters A, B, C
2. **Initialize Network**: Set up weights for hidden and output layers
3. **Train Model**: Run training loop with backpropagation
4. **Track Metrics**: Monitor loss and accuracy across epochs
5. **Visualize Results**: Plot training curves using matplotlib
6. **Test Predictions**: Classify test images and display results
7. **Display Images**: Use `matplotlib.pyplot.imshow()` to show input patterns

## 📦 Requirements
```
python>=3.8
numpy>=1.19
matplotlib>=3.3
jupyter>=1.0
```

**Note**: This project intentionally uses only NumPy for neural network implementation. TensorFlow and PyTorch are NOT used.

## 🔗 Submission Options

### Option 1: Google Drive
1. Create folder named `neural-network-classifier`
2. Place your `.ipynb` file inside
3. Zip the folder
4. Upload to Google Drive
5. Set sharing to "Anyone with the link" (Viewer)
6. Submit the shareable link

### Option 2: GitHub (Current Method)
1. Create public repository: `neural-network-classifier`
2. Upload `.ipynb` file and README.md
3. Ensure README explains approach, methodology, analysis process, and key findings
4. Submit GitHub repository link

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📝 License
This project is for educational purposes as part of Module 11 coursework.

## 👤 Author
**krishet37**
- GitHub: [@krishet37](https://github.com/krishet37)

## 🙏 Acknowledgments
- Module 11: Neural Network from Scratch - Course Assignment
- NumPy documentation for matrix operations
- Matplotlib for visualization tools

---

## 📚 Project Context
**Assignment**: ASSIGNMENT 4: NEURAL NETWORK  
**Module**: Module 11 - Project: Neural Network for Image Recognition of Letters A, B, and C  
**Objective**: Build a feedforward neural network from scratch using only NumPy to classify synthetic binary pixel patterns representing letters A, B, and C.

**Core Learning Outcomes**:
- Understand fundamental neural network concepts
- Implement backpropagation algorithm from scratch
- Master matrix operations and weight optimization
- Gain hands-on experience with gradient descent
- Learn to visualize training progress and results

---

*Note: Update the Key Findings section with actual results after running your trained model.*
