# SEDtools
A simple tool that read and plots outputs from either magphys/sed3fit.

Usage:
```python
import SEDtools
z_kind = 'spec' # Or 'phot'
code_used = 'magphys' # Or 'sed3fit'
SED = SEDtools.SEDProperties(path_to_sed, path_to_fit, z_kind, code_used)

# To plot the SED
fig = plt.figure(figsize=(10,5))
ax = plt.subplot(111)
title = 'whatever'
SED.plot_SED(ax, title)
```

See some examples in the notebook.
