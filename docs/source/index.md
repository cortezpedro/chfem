# `chfem` Documentation

```{toctree} 
:caption: Quick Start
:maxdepth: 3
placeholders/readme.md
```

```{toctree} 
:caption: Python API
:maxdepth: 3
placeholders/api.md
```

#### Simple example

```python
import chfem
import numpy as np

array = np.zeros((100, 100, 100))
array[30:70, 30:70, 30:70] = 255
keff = chfem.compute_property('conductivity', array, mat_props={255: 1, 0: 0.1}, direction='x', output_fields="cube")
```

```{toctree} 
:caption: Tutorial
:maxdepth: 1
tutorial.nblink
```
