# equivariant_model

To run, first:

```sh
cd example/copdms
```

Then for training: l=2; rank 2 Cartesian tensor; the bispectrum components, coordinates, and tensors filenames inside inp file; -tc for converting Cartesian to spherical tensors; -rc for rcut; -reg for regularization
```sh
python3 ../../src/regression.py -l 2 -in inp -r 2 -tc -rc 4.1 -reg 0.1
```

For test: similar to above but change the filenames inside inp to files containing the test set; -sf to skip fitting and read coefficient from coefficient.npy file written from the previous step
```sh
python3 ../../src/regression.py -l 2 -in inp -r 2 -tc -rc 4.1 -reg 0.1 -sf
```
