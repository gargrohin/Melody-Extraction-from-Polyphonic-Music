### Python files that were used for the final experiments.

Don't change their names. Many reference other files and import functions from them.

##### Brief description.

- `create.py` file uses a finction from `generate.py` to preprocess the raw `.wav` files. Converts them to their Power Spectrum.

- `nmf.py`: implementation of Non Negative Matrix Fatcorization algorithm, for source-filter NMF. 

- `my\_model.py`: implementation of the CRNN model using tensorflow and keras. Also has the data_generator and code for trainging the model.
