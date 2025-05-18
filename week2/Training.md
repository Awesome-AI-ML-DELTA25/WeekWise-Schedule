## What's training?
Training is the process of finding the parameters of the model that minimize the loss function. This is done by using an optimization algorithm, which is a method to find the minimum of a function. The most common optimization algorithm used in machine learning is **Gradient Descent**.

## The Procedure
Our aim is to find the parameters that minimize the loss function, but the obtained model must also be able to generalize well to unseen data. This is called **Generalization**. The process of training a model involves the following steps:
1. Split the data into training and testing sets. The training set is the one used to find the parameters of the model, while the testing set is used to evaluate the performance of the model, and act as a proxy for unseen data.
2. Choose a model and a loss function. The model is the class of functions we restrict ourselves to, and the loss function is the measure of distance between the true function and the approximated function.
3. Choose an optimization algorithm. The most common optimization algorithm used in machine learning is **Gradient Descent**.
4. Initialize the parameters of the model. This is usually done by randomly initializing the parameters.
5. Train the model by iteratively updating the parameters using the optimization algorithm.
6. Evaluate the performance of the model on the testing set. This is done by calculating the loss on the testing set.
7. If the performance is satisfactory, we can use the model to make predictions on unseen data. If not, we can go back to step 2 and try a different model or loss function.

**NEVER USE THE TESTING SET TO TRAIN THE MODEL.** The testing set is used to evaluate the performance of the model, and should not be used to find the parameters of the model. This is called **Data Leakage**, and can lead to overfitting, which is when the model performs well on the training set, but poorly on unseen data.

## The Optimization Algorithm
The most common optimization algorithm used in machine learning is **Gradient Descent**. The idea behind gradient descent is to iteratively update the parameters of the model in the direction of the negative gradient of the loss function. The gradient is a vector that points in the direction of the steepest increase of the function, and the negative gradient points in the direction of the steepest decrease. By moving in the direction of the negative gradient, we can find the minimum of the function.
The update rule for gradient descent is given by:
$$
\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)
$$
where $\theta_t$ is the vector of parameters at iteration $t$, $\eta$ is the learning rate, and $\nabla L(\theta_t)$ is the gradient of the loss function with respect to the parameters at iteration $t$. The learning rate is a hyperparameter that controls the step size of the update. A small learning rate will result in slow convergence, while a large learning rate may cause the algorithm to diverge.

### Hyperparameters
Hyperparameters are parameters that are not learned from the data, but are set before training the model. These include the learning rate, the number of iterations, the batch size, etc. The choice of hyperparameters can have a significant impact on the performance of the model. Hyperparameter tuning is the process of finding the best hyperparameters for a given model and dataset. This is usually done by using a validation set, which is a subset of the training set that is used to evaluate the performance of the model during training. We'll discuss hyperparameter tuning in detail later.

### Analogy
Let's demonstrate gradient descent with an analogy. Imagine you're in a dark night on a mountain, and you want to find the lowest point in the valley. You can't see anything, but you can feel the slope of the ground beneath your feet. You can use this information to take small steps in the direction of the steepest descent. By iteratively taking small steps in the direction of the steepest descent, you can find the lowest point in the valley. This is similar to how gradient descent works.
