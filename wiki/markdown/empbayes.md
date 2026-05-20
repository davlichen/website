
---
title: 'Empirical Bayes'
...

\newcommand{\N}{\mathbb{N}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\Q}{\mathbb{Q}}
\newcommand{\Z}{\mathbb{Z}}
\newcommand{\C}{\mathbb{C}}

\DeclareMathOperator{\Jac}{Jac}
\DeclareMathOperator{\Ker}{Ker}
\DeclareMathOperator{\mesh}{mesh}
\DeclareMathOperator{\Var}{Var}
\DeclareMathOperator{\Cov}{Cov}
\DeclareMathOperator{\hm}{hm}
\DeclareMathOperator{\sgn}{sgn}
\DeclareMathOperator{\regret}{regret}

\let\temp\phi
\let\phi\varphi
\let\varphi\temp

\newcommand{\pa}[1]{\left(#1\right)}
\newcommand{\bra}[1]{\left[#1\right]}
\newcommand{\cbra}[1]{\left\{#1\right\}}

\newcommand{\mat}[1]{\begin{matrix}#1\end{matrix}}
\newcommand{\pmat}[1]{\pa{\mat{#1}}}
\newcommand{\bmat}[1]{\bra{\mat{#1}}}

\newcommand{\pfrac}[2]{\pa{\frac{#1}{#2}}}
\newcommand{\bfrac}[2]{\bra{\frac{#1}{#2}}}
\newcommand{\psfrac}[2]{\pa{\sfrac{#1}{#2}}}
\newcommand{\bsfrac}[2]{\bra{\sfrac{#1}{#2}}}

## Parallel Statistical Problems

-------

### A Preliminary Example

This is essentially Robbins (1951).

Consider the decision problem of choosing $\theta \in \{-1, 1\}$ after observing $X \sim N(\theta, 1)$. The clear minimax decision rule is given by $\theta = \sgn(X)$.

Under some prior that assigns $P(\theta = 1) = \pi$, we have that the Bayes optimal rule is
$$
  \hat \theta_{\pi} = \sgn \left(X - \frac{1}{2} \log \left( \frac{1 - \pi}{\pi} \right)  \right)
$$
which incurs frequentist risk
$$
  R_\pi = P(\hat \theta_\pi \neq \theta) = \pi \Phi \left( \frac{1}{2} \log \left( \frac{1 - \pi}{\pi} \right) - 1\right) + 
  (1 - \pi) \Phi \left( -\frac{1}{2} \log \left( \frac{1 - \pi}{\pi} \right) - 1\right).
$$
Note that the minimax rule is equivalent to $\pi = \frac{1}{2}$ and has risk $R_{1/2} = \Phi(-1)$.

Now consider observations $X_1, \dots, X_n$ which obey $X_i \sim N(\theta_i, 1)$ under the loss $L(\hat \theta, \theta) = \frac{1}{n}\sum_{i=1}^n 1\{\hat \theta_i \neq \theta_i\}$ (which simply counts the proportion of entries in which we make an error). The minimax risk is again achieved by taking the sign component-wise, and suppose that we again have the prior which assigns $P(\theta_i = 1) = \pi$ independently. Again the Bayes optimal rule is still simple and is just the above applied component-wise.

Now suppose that we are willing to make the assumption that $P(\theta_i = 1) = \pi$ for *some* $\pi$, but one which is unknown. Then we set 
$$
  \hat \pi = \left(\frac{\frac{1}{n} \sum_{i=1}^n X_i + 1}{2} \land 1\right) \lor -1
$$
as (essentially) the method of moments estimator; we then take $\hat \theta_{EB} = \hat \theta_{\hat \pi}$ which has asymptotic risk equal to $R_{\pi}$ where here $\pi = \lim_{n \to \infty} \hat \pi$ is the "true" prior parameter.

Now we treat the parameters $\theta_i$ as fixed and translate the above estimator into this setting. Set $\pi_n = \frac{1}{n} \sum_{i=1}^n 1\{\theta_i = 1\}$, we have that (without the clipping) $\hat \pi \sim N(\pi_n, \frac{1}{4n})$, and so $\hat \theta_{EB}$ has risk approximately equal to $R_{\pi_n}$ for large $n$. Note that $R_{\pi_n} < R_{1/2}$, the minimax risk for any $\pi_n \neq 1/2$.

### The General Setup

Now consider the general setup of parameters $\theta_1, \theta_2, \dots, \theta_n, \dots$ and observations $X_1, X_2, \dots, X_n, \dots$ with each $X_i \sim P_{\theta_i}$ and either $\theta_i \sim G$ or $\theta_i$ some fixed constant. We will use $\underline \theta$ (and underlines in general) to denote the full vector of $\theta_i$. Finally, denote the parameter space as $\Omega$ and the space observations reside within as $\mathcal X$.

There are a couple of choices for assessing the performance of an estimator $\hat t: \mathcal X \to \Omega$ which we assume to depend only on $X_1, \dots, X_n$; we will also write $\underline t: \mathcal X^n \to \Omega^n$ for an estimator which maps observation vectors to parameter vectors.

_Def_: The **compound loss** is given as
$$
  R(\underline t, \underline \theta) = \frac{1}{n} \sum_{i=1}^n E[L((\underline t(X))_i, \theta_i)]
$$
whereas the **out-of-sample** loss is simply
$$
  E[L(\hat t(X_{n+1}, \theta_{n+1})]
$$
which in the case we take $\theta_i \sim G$.

_Def_: The **regret** of an estimator $\hat t$ is given by
$$
  \regret(\hat t) = E_G[L(\hat t(X_{n+1}), \theta_{n+1})] - R(G)
$$
where $R(G)$ is the Bayes risk under $G$.

_Def_: We say that an estimator $\underline t: \mathcal X^n \to \Omega^n$ is **equivariant** if $\underline t$ is invariant under permutation of the $(\theta_i, X_i)$; we say that it is **separable** if there is a univariate $t: \mathcal X \to \Omega$ such that $\underline t(\underline X) = (t(X_1), \dots, t(X_n))$.

If we consider $\theta_1, \dots, \theta_n$ to be fixed instead, we are interested in the quantity
$$
  R(\underline \theta) = \inf_{\underline t} \{ R(\underline t, \underline \theta) \}
$$
where the infimum is taken over all estimators $\underline t$ which are equivariant and separable.

**Theorem (Fundamental Theorem of Compound Decisions)**: Consider the empirical distribution
$$
  G_n(\underline \theta) = \frac{1}{n} \sum_{i=1}^n \delta_{\theta_i}.
$$
Then, we have that
$$
  R(G_n(\underline \theta)) = R(\underline \theta)
$$
where the LHS refers to the Bayes risk.

As a corollary, we have that the best equivariant and separable decision rule is the Bayes decision with respect to the prior of the empirical distribution.

<!-- ### $F$-Modeling and $G$-Modeling -->

<!-- In the above, we used "$F$-modeling," by which we mean we modeled the marginal $F_G(X)$ -->
