## Goal
This file explains the basic idea behind the objective of machine learning and what we're trying to achieve with it.

## Problem Statement
We have data given by a set of points in a space, and we want to find a function that maps these points to some output. The goal is to find a function that can generalize well to unseen data.

Formally, there's a hidden function $\tilde{f}$ that takes in input features $x$ and outputs a label $y$. We're given some $x_i$ and $y_i$ pairs, and we want to find a function $\hat{f}$ that approximates $\tilde{f}$ as closely as possible. 

What does it mean by "as closely as possible"? For this we need some measure of distance between the two functions. This function should take in two functions and return a number that represents how far apart they are, with closer functions returning smaller numbers. Such a function is denoted by $L(f, g)$, where $f$ is the true function and $g$ is the approximated function, and is called the **Loss**.

Now, since the true function is unknown, we can only use the data we have to approximate it. Instead of trying to find distance between the true function and the approximated function, we can find the distance between the approximated function and the data points we have. 

That is, now we define the distance between the approximated function and the original as follows:
$$
L(\tilde{f}, \hat{f}) = \frac{1}{N} \sum_{i=1}^N l(y_i, \hat{f}(x_i))
$$
where $l(y_i, \hat{f}(x_i))$ is called the **Loss Function**. This function takes in the true label $y_i$ and the predicted label $\hat{f}(x_i)$ and returns a number that represents how far apart they are.

Our goal is to find a function $\hat{f}$ that minimizes the loss incurred on the training data. This is called the **Empirical Risk Minimization** (ERM) principle. That is, we want:
$$
\hat{f} = \underset{{\hat{f} \in \text{Space of all Functions}}}{\arg \min} L(\tilde{f}, \hat{f}) = \underset{{\hat{f} \in \text{Space of all Functions}}}{\arg \min} \frac{1}{N} \sum_{i=1}^N l(y_i, \hat{f}(x_i))
$$

## The Issue
The space of all functions is forms a vectorspace, and is infinite dimensional. This suggests that there's no clean way to find the function that minimizes the loss.

## Solution: The Model
To solve this problem, we need to restrict the space of functions we can choose from, by making some assumptions about the function we want to find. This is called the **Model**.

A model is a class of functions that we restrict ourselves to. The choice of model depends on the problem we're trying to solve. And how the data looks like. For example, if we want a function that is simple to compute, we can use a linear model. Under the linear model assumption, we can write the function as:
$$
\hat{f}(x) = w^T x + b
$$
where $w$ is a vector of weights and $b$ is a bias term. This is the equation of a hyperplane in the input space. This restricts the space of functions we can choose from to a finite dimensional space, which is much easier to work with. The objective now is to find the weights and bias that minimize the loss function:
$$
\hat{w}, \hat{b} = \underset{{w, b} \in \mathbb{R}^n\times \mathbb{R}}{\arg \min} \frac{1}{N} \sum_{i=1}^N l(y_i, \hat{f}(x_i))
$$
where $l(y_i, \hat{f}(x_i))$ is the loss function.

Similarily, we can use other models like decision trees, neural networks, etc. The choice of model depends on the problem we're trying to solve and the data we have. We'll discuss the different models in detail later.

## Generalization
Let's assume that our model of choice, i.e., the class of functions we chose, is represented as follows:
$$
\mathcal{F} = \{f_{\theta} : \theta \in \mathbb{R}^d\}
$$
where $\theta$ is a vector of parameters that define the model. In the linear case, $\theta = (w, b)$.

In layman terms, the model is like a machine. The structure of the machine is fixed, but we can change the parameters of the machine to get different outputs. The goal is to find the parameters that minimize the loss function. Another way to think about it is that the model is a recipe, where the ingredients are fixed, but the brands/quality of the ingredients can be changed. The goal is to find the best mixture of ingredients that makes the best dish.
$$
\hat{f} = \underset{{f \in \mathcal{F}}}{\arg \min} L(\tilde{f}, f) = \underset{{f \in \mathcal{F}}}{\arg \min} \frac{1}{N} \sum_{i=1}^N l(y_i, f(x_i))
$$
Alternatively, we can write this as:
$$
\hat{\theta} = \underset{{\theta \in \mathbb{R}^d}}{\arg \min} L(\tilde{f}, f_{\theta}) = \underset{{\theta \in \mathbb{R}^d}}{\arg \min} \frac{1}{N} \sum_{i=1}^N l(y_i, f_{\theta}(x_i))
$$
and the approximated function is given by:
$$
\hat{f} = f_{\hat{\theta}}
$$
where $\hat{\theta}$ is the vector of parameters that minimize the loss function.

## Examples of Models
Models are a way to hypothesize, what a certain function/system looks like. They help in analysis of complex situation. You must've seen several examples of models in your life. For example, any place where you make a simplyfing assumption about a system, is a inherently adopting a model (Bohr's/Dalton's model of atom, Ignoring friction in a system, etc.).

In the context of machine learning, models are used to hypothesize what the function looks like. For example, if we have a set of points in a 2D space, we can use a linear model to fit a line through the points. This is called **Linear Regression**. The model assumes that the relationship between the input and output is linear, and tries to find the best line that fits the data.

Similarily, a decision tree is a model that assumes that the data can be split into different regions based on some features. The model tries to find the best splits that minimize the loss function.

A neural network is a model that assumes that the data can be represented as linear combinations of features followed by non-linear transformations. The model tries to find the best weights and biases that minimize the loss function.
