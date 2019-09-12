---
layout: post
title: k-Nearest Neighbors and the Curse of Dimensionality
date: 2019-09-05
categories: [Machine Learning, Mathematics]
image: /assets/images/ml_course/knn_header.jpg
excerpt: The k-Nearest Neighbors algorithm is an easy to understand algorithm that can be used for both classification and regression.
published: false
---

> This is the second part of a multi-part series introducing machine learning, with a focus on supervised learning. Check out the first part [here]() if you're getting lost with the terminology or notation.

The k-Nearest Neighbors algorithm is an easy to understand algorithm that can be used for both classification and regression. It is one of the older machine learning algorithms still in use, as it was first published in [1967 by Cover and Hart](https://ieeexplore.ieee.org/document/1053964).

## Informal Definition

The idea is quite simple. We assume that points that are close together will have the same label. If that assumption is true, the following should give pretty decent predictions for new inputs. For a given set of features $$\textbf{x}$$, we

1. Find the $$k$$ data points in our training set $$D$$, which are the closest to $$\textbf{x}$$.
2. With these $$k$$ nearest data points:
   1. Classification: we return the most common label.
   2. Regression: We order the points by distance and do a kind of local regression.

That's it! Although it's pretty easy to explain there are some interesting details to dive into for this algorithm. How do we find the points closest to our $$\textbf{x}$$? How do we choose our $$k$$? In what situations will the algorithm perform well, and when won't it? What kind of assumptions are we making on our space of possible models? We'll diving into exactly these kinds questions in this post.

## Formal (and confusing) Definition

Just in case you're curious, we'll be stating the rigorous mathematical definition here. Although much more complicated than the simple algorithm stated above, it can be good to know in case you're wondering about certain details. Feel free to skip over this section if you're not in the mood for math.

Given a point $$\textbf{x} \in \mathbb{R}^d$$, we'll first define what the set $$S_\textbf{x}$$ of it's nearest neighbors looks like. We want it to satisfy
$$
S_\textbf{x} \subseteq D \text{ such that } |S_{\textbf{x}}| = k
$$

and

$$
\text{For all $(\textbf{x}',y') \in D\setminus S_\textbf{x}$}\quad \text{we have}\quad \text{dist}(\textbf{x}, \textbf{x}')\geq \max_{(\textbf{x}'',y'') \in S_\textbf{x}}\text{dist}(\textbf{x}, \textbf{x}'')
$$

So let's take a little time to unpack what's written here. The first statement says we want a subset of points in $D$ with size $k$. The interesting side is the second part, we want each point in this set to have a smaller distance to $$\textbf{x}$$ the every other point in $D$ not in $$S_\textbf{x}$$. Note that we write $\text{dist}(\cdot)$ since this doesn't have to be the euclidean distance, but anything which is a valid [metric](https://en.wikipedia.org/wiki/Metric_(mathematics)).

We can now write the classifier in terms of the set $$S_\textbf{x}$$:

$$
h(\textbf{x}) := \text{mode}(\{y' : (\textbf{x}',y') \in S_\textbf{x}\})
$$

## Impact of $k$

The parameter $k$ can be chosen to be whatever we want, and has a large impact on how the algorithm works. In general you want to choose your $k$ through [hyperparameter optimization](https://en.wikipedia.org/wiki/Hyperparameter_optimization), but it's still useful to know what kind of impact smaller or larger values for $k$ have on what the algorithm is actually doing.

- Extreme k values (1, and infinity)
- k controls model complexity, so large k is simple (low variance, high bias) and small k is complex (high variance, low bias).
- Compare this to some simple-to-visualize dataset.
- K can also be viewed as a smoothing parameter, to remove noise.

## Choice of Distance Functions (Metrics)

It should now be clear that we're making an implicit assumption about our data, namely that points which are close together have the same labels. What makes this interesting, is that we are free to choose how we define "closeness" up to a certain degree, with our choice of the distance function the algorithm will use.

As an example, a commonly used family of metrics is the *Minkowski distance*, which for a given $$p$$ has the form

$$
\text{dist}(\textbf{x}, \textbf{z}) = \left(\sum_{i=1}^d |x_i - z_i|^p\right)^{1/p}.
$$

This contains many familiar metrics you might know. For $$p=1$$ this becomes the Manhattan or taxicab metric, for $$p=2$$ it becomes the Euclidean norm, and for $$p \rightarrow \infty$$ it becomes the max norm.

The Minkowski distance works great for continuous variables, but in the case of discrete variables you might want to look at something like the [Hamming distance](Hamming distance). An even more complex case is when you have a mix of continuous and discrete features. While Euclidean will work with such datasets, there are more complex methods like the [Mahalanobis distance](https://www.jstor.org/stable/2676979?seq=5#metadata_info_tab_contents) which should work better. Scikit-learn has a [list of available metrics](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html).

The choice of metric is so important that there's even a whole field focused on the learning of optimal metrics called [Metric/Similary Learning](https://en.wikipedia.org/wiki/Similarity_learning), and is considered a subfield of machine learning.

It's hard to give simple advice for what the correct metric is for your specific problem, but the nice thing about the Minkowski distance is that it has a single parameter which we can tune as a model hyper-parameter, and will work reasonably well on most datasets.

## Curse of Dimensionality

As we mentioned earlier, the k-nearest neighbors algorithm makes the assumption that points close together have the same label. Things get difficult when the dimension of our space becomes large, since as we'll show in a moment, samples drawn from a probability distribution in a high-dimensional space tend to always be far apart.

For this example we'll be considering a uniform distribution on a d-dimensional unit cube $$[0,1]^d$$, and an inner cube $$l_{\varepsilon,d}=[\varepsilon, 1 - \varepsilon]^d$$. See below for an illustration:

{% 
    include image.html 
    url= "http://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/images/c2/note2_3.png"
    description="Here we see the impact of choosing our hypothesis space, also known as the bias-variance tradeoff. If our space is too large (picture on right) we get fantastic results on our training set, but horrible generalization. If our space is too small (picture on left), we get bad results on the training set, and bad generalization."
    style="max-height: 300px;"
%}

Let's say we randomly sample a point $$X$$ (uniformly) in this large hypercube, and look at the probability that our new point is inside $$l_{\varepsilon,d}$$ when $$\varepsilon$$ is close to 1. In the 1-dimensional case, this is $$P(X \in l_{\varepsilon,1}) = 1 - 2\varepsilon$$. Since each dimension of the uniform distribution is independent, we can write

$$
P(X \in l_{\varepsilon,d}) = (1 - 2\varepsilon)^d.
$$

Let's look at some numbers to see, if we take $$\varepsilon = 0.05$$, how the probability $$P(X \in l_{\varepsilon,d})$$ change as the dimension increases. 

Here's a table with a few values:

| Dimension | $$P(X \in l_{\varepsilon,d})$$ |
| --------- | ------------------------------ |
| 1         | $$0.9$$                        |
| 10        | $$0.349$$                      |
| 100       | $$2.66\cdot10^{-5}$$           |
| 1000      | $$1.7\cdot 10^{-46}$$          |

As you can see, as the dimension increases, most of the points end up getting sampled at the edges of the cube. While this seems counter-intuitive, one can show that in very high dimensions, most of the space is concentrated at the edges of the cube.

This is all nice and well, but why should we care? To understand this lets take a look at the figure below

{% 
    include image.html 
    url= "http://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/images/c2/cursefigure.png"
    description="Here we see the impact of choosing our hypothesis space, also known as the bias-variance tradeoff. If our space is too large (picture on right) we get fantastic results on our training set, but horrible generalization. If our space is too small (picture on left), we get bad results on the training set, and bad generalization."
%}

What this shows is that as the dimension of our space increases, the distances of these points get pushed together. This means that our assumption of *"points close together have the same label"* doesn't work too well if all the points are equally close together.

Given the curse of dimensionality, is our k-nearest neighbors algorithm still useful for most real-world datasets? Or are we stuck with only using datasets of only a few features? Not all is lost, as we'll explain in the next section.

## Data in Lower Dimensional Manifolds

In the examples above, we showed that when we randomly sample our data, distances lose their meaning in a sense. The problem there is that are data was sampled in an independent way, and it didn't contain much structure. 

In most machine learning scenarios, our data hopeful contains patterns and structure, which can be viewed as lower-dimensional manifolds within our d-dimensional feature space. This is called the [Manifold Hypothesis](https://deepai.org/machine-learning-glossary-and-terms/manifold-hypothesis) in machine learning. Christopher Olah has a [great post](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/) on this in the context of neural networks. If you don't know his blog yet ([colah.github.io](colah.github.io)), definitely check it out.

See the picture below for a visual explanation.

{% 
    include image.html 
    url="/assets/images/ml_course/manifold.png"
    description="Here we see the impact of choosing our hypothesis space, also known as the bias-variance tradeoff. If our space is too large (picture on right) we get fantastic results on our training set, but horrible generalization. If our space is too small (picture on left), we get bad results on the training set, and bad generalization."
%}

Another way of thinking of this, consider a situation where we're trying to represent human faces. Say the the pictures are $$256\times256$$ pixels with B/W values. Even though such pictures might contains 65,536 features, we might be able to sufficiently describe them with say 50 properties such as "male/female", "light/dark hair", "position of nose w.r.t. eyes", etc. This is a case where there is a lower-dimensional sub-manifold which sufficiently describes the data.

These manifolds can take many different forms, and we need to find a way to expose such structure to our distance function. This can be done using feature selection (removing non-informative features), or dimension-reduction techniques like [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis), [linear descriminant analysis](https://www.wikiwand.com/en/Linear_discriminant_analysis) or [t-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding). 

## Appendix: Finding Nearest Neighbors

Although we didn't spend much time on this, the problem of finding the k-nearest neighbors for large datasets is not a trivial problem. The naive approach to finding the nearest neighbor is to some point $\textbf{x}$ is to first calculate the distance $$\text{dist}(\textbf{z},x)$$ to every point $$\textbf{z} \neq \textbf{x}$$. This would take huge amounts of work every time we want to find a neighbor.

The two algorithms which are used in scikit-learn are [K-D Trees](https://en.wikipedia.org/wiki/K-d_tree) and [Ball Trees](https://en.wikipedia.org/wiki/Ball_tree), which both split up the search space into smaller parts, and inform us where to search for neighbors first to speed things up. The linked wikipedia pages give a fairly decent explanation of both algorithms.

Although when creating a machine learning algorithm you'll often not have to worry about these techniques, there might be other problems where you'll need an algorithm like this. 

