# Notes

- Cyclical learning rates? Increase initial learning rate?
- Try different size of hidden layers, different optimizer etc.
- Refactor `verify_povm` to calculate all inner products and assert that max
relative diff is less than tolerance.
- Add a small perturbation to the input vector, and combine the different
results in a mini batch, perhaps of size d?
