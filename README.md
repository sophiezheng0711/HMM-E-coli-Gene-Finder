# CS4775 Final Project

*A hidden Markov model that finds genes in E.coli DNA* (Krogh et al) reimplementation with both simple and complex models.

# Run Instructions

```
python3 main.py -m simple
python3 main.py -m complex
```

# Simple Intergenic Model

If any changes were made to the structure of the simple model, run this:

```
python3 simple_model_constructor.py
```

This will regenerate the transition probabilities, emission probabilities, as well as the structure of the states.

# Complex Intergenic Model

If any changes were made to the structure of the complex model, run this:

```
python3 complex_em.py
```
