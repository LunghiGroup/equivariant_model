# equivariant_model

The Python code for equivariant ridge regression model. For more information:
* Predicting tensorial molecular properties with equivariant machine-learning models, arXiv:2202.01449 [cond-mat.mtrl-sci], 2022

This program is free software: you can redistribute it and/or modify it. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. Please note that the current implementation is not yet optimized.

### Prerequisites
The code requires Python 3 and the following Python libraries: ase and numpy.

## Manual
There are 2 simple steps to run the model:
1. Training the model with a training set, this step will save the model coefficients in `coefficients.npy`, output the prediction spherical tensor values along with the corresponding reference values in `tensors.out`, and print the RMSE.
2. The model performance can be tested with the option `-sf` or `--skipfit`. This will make the code read the model's coefficients from the `coefficients.npy` generated in the previous step. This step will output output the prediction spherical tensor values along with the corresponding reference values of the test set and print out the test RMSE.

## Example
3 sets of molecules are provided along with this code in the `example/` directory. The following is an example of running the model for CoL<sub>2</sub>. The same procedure can be applied to all other examples.

### CoL<sub>2</sub>
First go to the example directory of CoL<sub>2</sub>

```sh
cd example/copdms
```

Then, to train the model with the training set: 
```sh
python3 ../../src/regression.py -l 2 -in inp -r 2 -tc -rc 4.1 -reg 0.1
```

The `inp` file (`-in inp` or `--input inp`) contains the names of the files with the bispectrum components, the coordinates, and the tensors of all configurations in the training set. As the target property is the D tensor (a rank 2 Cartesian tensor: `-r 2` or `--rank 2`), the corresponding spherical tensor order is l=2 (`-l 2` or `--lorder 2`). If the tensor file contains Cartensian tensors, `-tc` or `--tensconv` is needed to convert the Cartesian tensor to spherical tensor of the correct order l. Hyperparameters rcut and regularization value can be specified with `-rc` or `--rcut` and `-reg` or `--regularize`, respectively.


To test the performance of the model, the command is similar to the previous step but the option `-sf` or `--skipfit` is necessary to read the coefficient file from the previous step instead of fitting the model again. Additionally, the file names in `inp` needs to be changed to point to the correct files containing the test set data.
```sh
python3 ../../src/regression.py -l 2 -in inp -r 2 -tc -rc 4.1 -reg 0.1 -sf
```

The hyperparameters can be optimized by varying the values and testing the model's performance with a validation set.
