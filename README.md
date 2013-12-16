CHIMEpulsar
===========

Scripts for looking at pulsars in CHIME data.  There is some (questionably) useful information in the docstrings.  I'll give a few usage examples here.

In all of these cases let's assume this module has been imported like so:
>> import CHIMEpulsar.CHIMEdata as ch

If you have a set of contiguous hdf5 files, put the filenames in a sorted list (I recommend using the "glob" command in the "glob" module) and read them in like so:

```python
from glob import glob
chime_fpaths = glob("filename_root.h5.*")
chime_fpaths.sort()
chd = ch.CHIMEdata(chime_fpaths, datachans=0)
```

The data is stored in chd.data as a 2D array over frequency and time.

The "datachans" argument can be a single number of a list of numbers.  It refers to the correlator channel(s) read in from the data.  At this time, it's assumed that autocorrelations are being used and multiple channels are simply added together.

You can save the data you've read in to a smaller, faster-loading .npy file:
```python
chd.save_data(savedata_filename)
```

If you have already created such a file, then instead of reading in from a list of hdf5 files, you can read in the .npy file:
```python
chd = ch.CHIMEdata(savedata_filename)
```

The built-in methods "auto_freq_mask" and "auto_time_mask" will mask the data where there appear to be large spikes either over a whole frequency channel or over short durations in individual frequency channels.  The method "pad_missing_samples" will look for large jumps in the timestamps and put masked dummy into the appropriate places.  If there are missing samples and this is not done, pulsars will not fold properly.  Finally, "replace_masked_times_with_noise" will remove masking that does not cover whole frequency channels and replace those points with white noise.  Keeping masked times causes problems when dedispersing.

So a sensible thing to do after reading in the data is to run all four of these in succession:
```python
chd.auto_freq_mask()
chd.auto_time_mask()
chd.pad_missing_samples()
chd.replace_masked_times_with_noise()
```

If you want to look at the spectrum for a 1D time series dedispersed to some DM value:
```python
chd.calc_spectrum(DM)
chd.spectrum.show()
```

(You may need to follow this and all other '.show()' commands with ch.plt.show()

If you want to fold a pulsar with period p0 and dispersion measure DM using all of the (unmasked) data:
```python
chd.fold_pulsar(p0, DM, nbins)
chd.profile.show()
```
where 'nbins' is the number of bins across your folded profile.

If you want the profile over time or over frequency in a waterfall plot,
```python
chd.calc_phase_vs_time(p0, DM, nbins, nints)
chd.phase_vs_time.show()
chd.calc_phase_vs_freq(p0, DM, nbins, nsubs)
chd.phase_vs_freq.show()
```
where 'nints' and 'nsubs' are the number of subintegrations and sub-bands to divide the profile across respectively.  In the case of calc_phase_vs_freq, you can also set "dedisp=False" in order to see the dispersion curve.
