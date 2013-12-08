import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from glob import glob
import os, sys
import scipy.signal as sig
import pandas as PD

class CHIMEdata:
    """
    CHIME frequency and time data with some pulsar-related methods.

    'data' should either be a single .npy file or a list of raw CHIME files.

    'datachan' should (for now) be 0, 1, or 8:
      0: (0, 0)
      1: (0, 1)
      8: (1, 1)
    """
    def __init__(self, datafiles, datachan=0):
        # If 'data' is a string, check if it ends with .npy
        if isinstance(datafiles, str):
            if datafiles[-4:] == '.npy':
                self.data = np.load(datafiles)
            else:
                # Make it a list so it gets caught by the next if statement
                datafiles = [datafiles]

        if not isinstance(datafiles, str):
            chunks = []
            for datafile in datafiles:
                print "Loading %s..." % datafile
                dat = h5.File(datafile, mode='r')
                vis_sep = dat['vis'].value
                vis = vis_sep['real'] + 1.j*vis_sep['imag']
                chunks.append(vis[:,datachan,:])
                dat.close()
                print "Done."
            self.data = np.abs(np.concatenate(chunks, axis=0))

        self.data = np.ma.array(self.data,
                                mask=np.zeros(self.data.shape, dtype=bool))

        self.nfreq = self.data.shape[1]
        self.highfreq = 800.
        self.lowfreq = 400.
        self.freqs = np.linspace(self.highfreq, self.lowfreq, self.nfreq)\
            - (self.highfreq - self.lowfreq)/self.nfreq/2.

        self.nsamp = self.data.shape[0]
        self.tres = 0.01 # seconds
        self.times = np.linspace(0., self.nsamp*self.tres, self.nsamp,
                                 endpoint=False)

        # and some placeholders for things that can be calculated if desired
        self.tseries = None
        self.spectrum = None
        self.profile = None
        self.phase_vs_freq = None
        self.phase_vs_time = None

        self.detailed_mask = np.zeros(self.data.shape, dtype=bool)

    def auto_detailed_mask(self, threshold=5.):
        data = self.data.data
        channel_medians = np.median(data, axis=0)
        abs_devs_from_med = np.abs(data - channel_medians)
        channel_mads = np.median(abs_devs_from_med, axis=0)
        return abs_devs_from_med > threshold*channel_mads

    def zap_chans(self, first, last, samp1=None, samp2=None, unzap=False):
        """
        for detailed mask
        """
        if samp1 is None: samp1 = 0
        if samp2 is None: samp2 = self.nsamp - 1
        if not unzap:
            self.detailed_mask[samp1:(samp2+1), first:(last+1)] = True
        else:
            self.detailed_mask[samp1:(samp2+1), first:(last+1)] = False

    def show_data_w_mask(self):
        """
        for detailed mask
        """
        data = np.ma.array(self.data.data, mask=self.detailed_mask)
        plt.imshow(data, aspect='auto', cmap=plt.cm.Greens, interpolation='nearest')

    def replace_masked_times_with_noise(self):
        """
        for detailed mask
        """
        data = np.ma.array(self.data.data, mask=self.detailed_mask.copy())
        actual_mask = np.zeros(data.shape, dtype=bool)
        for ii in range(self.nfreq):
            chan_mask = self.detailed_mask[:,ii].copy()
            n_masked_bins = len(chan_mask.nonzero()[0])
            if self.nsamp - n_masked_bins:
                chan_mean = np.mean(data[:,ii])
                chan_std = np.std(data[:,ii])
                noise = np.random.normal(chan_mean, chan_std, n_masked_bins)
                data[self.detailed_mask[:,ii],ii] = noise
            else:
                actual_mask[:,ii] = True
            sys.stdout.write("\rProgress: %-5.2f%%" %\
                (100.*float(ii+1)/self.nfreq))
            sys.stdout.flush()
        self.data = np.ma.array(data.data, mask=actual_mask)

    def dm_delays(self, dm, f_ref):
        """
        Returns array of delays in seconds corresponding to frequencies
         in self.freqs.
        """
        return 4.148808e3 * dm * (self.freqs**(-2) - f_ref**(-2))

    def dedisperse(self, dm, **kwargs):
        """
        kwargs:
          start_chan: fold using data at or above this channel index
          end_chan: fold using data below this channel index
          start_samp: fold using data at or above this sample index
          end_samp: fold using data below this sample index
          (By default, all of the above encompass the entire data set.)

          no_save: if this is True, return output but don't store it in object
          (By default, it is False)
        """
        if kwargs.has_key('no_save'): no_save = kwargs['no_save']
        else: no_save = False
        if kwargs.has_key('start_chan'): start_chan = kwargs['start_chan']
        else: start_chan = 0
        if kwargs.has_key('end_chan'): end_chan = kwargs['end_chan']
        else: end_chan = self.nfreq
        if kwargs.has_key('start_samp'): start_samp = kwargs['start_samp']
        else: start_samp = 0
        if kwargs.has_key('end_samp'): end_samp = kwargs['end_samp']
        else: end_samp = self.nsamp
        data = self.data[start_samp:end_samp, start_chan:end_chan]
        freqs = self.freqs[start_chan:end_chan]
        nsamp = end_samp - start_samp
        nfreq = end_chan - start_chan
        k_flt = self.dm_delays(dm, freqs[0])[start_chan:end_chan]/self.tres
        k = (np.round(k_flt) + 0.1).astype(int)
        tseries = np.zeros(nsamp, dtype=float)
        for ll in range(nfreq):
            if not data.mask[:,ll][0]:
                tseries += np.roll(data[:,ll], k[ll])
        if not no_save: self.tseries = TimeSeries(tseries[np.max(k):], dm)
        return tseries[np.max(k):]

    def fold_pulsar(self, p0, dm, nbins=32, **kwargs):
        """
        kwargs:
          start_chan: fold using data at or above this channel index
          end_chan: fold using data below this channel index
          start_samp: fold using data at or above this sample index
          end_samp: fold using data below this sample index
          (By default, all of the above encompass the entire data set.)

          f_ref: reference frequency to dedisperse to
          (By default, it is the first channel of the frequency range used.)

          no_save: if this is True, return output but don't store it in object
          (By default, it is False)
        """
        if kwargs.has_key('no_save'): no_save = kwargs['no_save']
        else: no_save = False
        if kwargs.has_key('start_chan'): start_chan = kwargs['start_chan']
        else: start_chan = 0
        if kwargs.has_key('end_chan'): end_chan = kwargs['end_chan']
        else: end_chan = self.nfreq
        if kwargs.has_key('start_samp'): start_samp = kwargs['start_samp']
        else: start_samp = 0
        if kwargs.has_key('end_samp'): end_samp = kwargs['end_samp']
        else: end_samp = self.nsamp
        data = self.data[start_samp:end_samp, start_chan:end_chan]
        data -= running_mean(data)
        freqs = self.freqs[start_chan:end_chan]
        times = self.times[start_samp:end_samp]
        if kwargs.has_key('f_ref'): f_ref = kwargs['f_ref']
        else: f_ref = freqs[0] 
        profile = np.zeros(nbins, dtype=float)
#        counts = np.zeros(nbins, dtype=float)
        ddtimes = times.repeat(len(freqs)).reshape(data.shape)\
            - self.dm_delays(dm, f_ref)[start_chan:end_chan]
        phases = ddtimes / p0 % 1.
        which_bins = (phases * nbins).astype(int)
        for ii in range(nbins):
            vals = data[which_bins == ii]
#            counts[ii] = len((~vals.mask).nonzero()[0])
#            profile[ii] += vals.sum()
            profile[ii] += vals.mean()
#        return profile/counts
        if not no_save: self.profile = Profile(profile, p0, dm, nbins)
        return profile

    def calc_phase_vs_freq(self, p0, dm, nbins=32, nsubs=32, dedisp=True,
                           **kwargs):
        """
        kwargs:
          start_samp: fold using data at or above this sample index
          end_samp: fold using data below this sample index
          (By default, all of the above encompass the entire data set.)

          no_save: if this is True, return output but don't store it in object
          (By default, it is False)
        """
        if kwargs.has_key('no_save'): no_save = kwargs['no_save']
        else: no_save = False
        if kwargs.has_key('start_samp'): start_samp = kwargs['start_samp']
        else: start_samp = 0
        if kwargs.has_key('end_samp'): end_samp = kwargs['end_samp']
        else: end_samp = self.nsamp
        low_f = self.freqs[-1]
        high_f = self.freqs[0]
        waterfall = []
#        top_freqs = []
        for ii in range(nsubs):
            start_chan = ii*(self.nfreq/nsubs)
            end_chan = (ii+1)*(self.nfreq/nsubs)
            seg_freqs = self.freqs[start_chan:end_chan]
#            top_freqs.append(seg_freqs[0])
            if dedisp: f_ref = self.freqs[0]
            else: f_ref = seg_freqs[0]
            row = self.fold_pulsar(p0, dm, nbins,\
                start_samp=start_samp, end_samp=end_samp,
                start_chan=start_chan, end_chan=end_chan, f_ref=f_ref,\
                no_save=True)
#            row -= np.median(row)
#            row /= np.std(row)
            waterfall.append(np.tile(row, 2))
        waterfall = np.array(waterfall)
        if not no_save:
            self.phase_vs_freq = PhaseVSFreqPlot(waterfall,\
                low_f, high_f, p0, dm, dedisp)
        return waterfall

    def calc_phase_vs_time(self, p0, dm, nbins=32, nints=32, **kwargs):
        """
        kwargs:
          no_save: if this is True, return output but don't store it in object
          (By default, it is False)
        """
        if kwargs.has_key('no_save'): no_save = kwargs['no_save']
        else: no_save = False
        start_t = self.times[0]
        end_t = self.times[-1]
        waterfall = []
        for ii in range(nints):
            start_samp = ii*(self.nsamp/nints)
            end_samp = (ii+1)*(self.nsamp/nints)
            row = self.fold_pulsar(p0, dm, nbins,\
                start_samp=start_samp, end_samp=end_samp,\
                no_save=True)
#            row -= np.median(row)
#            row /= np.std(row)
            waterfall.append(np.tile(row, 2))
        waterfall = np.array(waterfall)
        if not no_save:
            self.phase_vs_time = PhaseVSTimePlot(waterfall,\
                start_t, end_t, p0, dm)
        return waterfall

    def mask_freqs(self, channel_ranges):
        for item in channel_ranges:
            self.data.mask[:, item[0]:(item[1]+1)] = True

    def clip_times(self, sample_ranges):
        for item in sample_ranges:
            before = self.data[item[0]-1]
            after = self.data[item[1]+1]
            if item[0] == 0: before = after
            if item[1] == self.nsamp - 1: after = before
            self.data[item[0]:(item[1]+1)] = 0.5*(before + after)

    def calc_spectrum(self, dm, N=None, whiten=False, **kwargs):
        """
        Not a lot of thought was put into the whitening, and it may be
        pretty sketchy.

        kwargs:
          start_chan: fold using data at or above this channel index
          end_chan: fold using data below this channel index
          start_samp: fold using data at or above this sample index
          end_samp: fold using data below this sample index
          (By default, all of the above encompass the entire data set.)

          no_save: if this is True, return output but don't store it in object
          (By default, it is False)
        """
        if kwargs.has_key('no_save'): no_save = kwargs['no_save']
        else: no_save = False
        if kwargs.has_key('start_chan'): start_chan = kwargs['start_chan']
        else: start_chan = 0
        if kwargs.has_key('end_chan'): end_chan = kwargs['end_chan']
        else: end_chan = self.nfreq
        if kwargs.has_key('start_samp'): start_samp = kwargs['start_samp']
        else: start_samp = 0
        if kwargs.has_key('end_samp'): end_samp = kwargs['end_samp']
        else: end_samp = self.nsamp
        tseries = self.dedisperse(dm, no_save=True,\
            start_chan=start_chan, end_chan=end_chan,\
            start_samp=start_samp, end_samp=end_samp)
        nyquist_f = 0.5/self.tres # in Hz
        if N is None: N = len(tseries) - (50+50)
#        fft = np.fft.rfft(tseries - np.mean(tseries), n=N)
        fft = np.fft.rfft((tseries - running_mean(tseries, 50))[50:-50], n=N)
        fft_amp = np.abs(fft)
        freq_axis = np.linspace(0, nyquist_f, len(fft_amp))
        if whiten:
            split = np.array_split(fft_amp, 100)
            for ii in range(len(split)):
                split[ii] -= np.median(split[ii])
                split[ii] /= np.std(split[ii])
            fft_amp = np.concatenate(split)
        if not no_save: self.spectrum = Spectrum(fft_amp, freq_axis, dm, whiten)
        return freq_axis, fft_amp

    def save_data(self, out_fname):
        """
        Saves the data without the mask.
        """
        np.save(out_fname, self.data.data)

class PhaseVSFreqPlot:
    def __init__(self, data, low_f, high_f, p0=None, dm=None, dedisp=None):
        self.data = data
        self.low_f = low_f
        self.high_f = high_f
        self.p0 = p0
        self.dm = dm
        self.dedisp = dedisp
    def show(self):
        plt.imshow(self.data, aspect='auto', cmap=plt.cm.Greens,
                   interpolation='nearest', origin='upper',
                   extent=(0., 2., self.low_f, self.high_f))
        plt.xlabel("phase")
        plt.ylabel("observing frequency (MHz)")

class PhaseVSTimePlot:
    def __init__(self, data, start_t, end_t, p0=None, dm=None):
        self.data = data
        self.start_t = start_t
        self.end_t = end_t
        self.p0 = p0
        self.dm = dm
    def show(self):
        plt.imshow(self.data, aspect='auto', cmap=plt.cm.Greens,
                   interpolation='nearest', origin='lower',
                   extent=(0., 2., self.start_t, self.end_t))
        plt.xlabel("phase")
        plt.ylabel("observing time (s)")

class TimeSeries:
    def __init__(self, data, dm=None):
        self.data = data
        self.dm = dm
    def show(self):
        plt.plot(self.data)

class Spectrum:
    def __init__(self, data, freq_axis, dm=None, whitened=None):
        self.data = data
        self.freq_axis = freq_axis
        self.dm = dm
        self.whitened = whitened
    def show(self):
        plt.plot(self.freq_axis, self.data)
        plt.xlabel("frequency (Hz)")
    def show_power(self):
        plt.plot(self.freq_axis, self.data**2)
        plt.xlabel("frequency (Hz)")

class Profile:
    def __init__(self, data, p0=None, dm=None, nbins=None):
        self.data = data
        self.p0 = p0
        self.dm = dm
        self.nbins = nbins
    def show(self):
        plt.plot(np.linspace(0, 2, 2*len(self.data), endpoint=False),
                 np.tile(self.data, 2))
        plt.xlabel("phase")

def running_mean(arr, radius=50):
    n = radius*2+1
    try: mask = arr.mask
    except: pass
    padded = np.concatenate((arr[1:radius+1][::-1], arr,\
        arr[-radius-1:-1][::-1]))
    ret = np.cumsum(padded, axis=0, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    try: return np.ma.array(ret[n-1:]/n, mask=mask)
    except: return ret[n-1:]/n

chime_fpaths = glob("../20131208T070336Z/20131208T070336Z.h5.*")
#chime_fpaths = glob("20131121T071019Z/20131121T071019Z.h5.0*")
chime_fpaths.sort()

data_fname = "em00_20131208T070336Z.npy"
#data_fname = "em00_20131121T071019Z.npy"
#data_fname = "em01_20131121T071019Z.npy"
#data_fname = "em11_20131121T071019Z.npy"

DM = 26.833

#p0 = 0.7145196997258 <-- 10 years ago
p0 = 0.7145214963305137 * (1. + 8.77e-6)
# (the correction comes from V_lsr which I think is -2.63 km/s and probably
# makes no difference)

zapfreqs = [
    (0,0),
    (268,269),
    (553,599),
    (630,660),
    (676,691),
#    (754,767),
    (754,798),
#    (784,798),
    (806,813),
    (846,846),
    (855,859),
    (868,868),
    (873,882),
    (889,890),
    (895,895),
    (974,975),
    (977,978),
    (988,1023),
]

zaptimes = [
    (0,0),
    (4006,4006),
    (4261,4261),
#    (8346,8407),
    (8150,8550),
    (12215,12215),
    (12585,12585),
#    (13050,13111),
    (12850,13450),
    (14425,14425),
    (20809,20809),
    (22297,22297),
]

if os.path.exists(data_fname):
    chd = CHIMEdata(data_fname)

else:
    chd = CHIMEdata(chime_fpaths)
    chd.save_data("../" + data_fname)

#chd.detailed_mask = np.load("detailed_mask.npy")
#chd.replace_masked_times_with_noise()

#chd.mask_freqs(zapfreqs)
#chd.clip_times(zaptimes)

"""
def norm_data(data, windowsize=100):
    new_data = data.copy()
    nbins = data.shape[0]
    bins = np.array(range(nbins))
    lows = bins - windowsize/2
    highs = bins + windowsize/2
    highs[np.where(lows < 0)] -= lows[np.where(lows < 0)]
    lows[np.where(lows < 0)] = 0
    lows[np.where(highs > nbins)] -= (highs[np.where(highs > nbins)] - nbins)
    highs[np.where(highs > nbins)] = nbins
    for ii in bins:
        new_data[ii] -= data[lows[ii]:highs[ii]].mean(axis=0)
    return new_data
"""

"""
# assuming fwhm of about 4 degrees, which seems right for the big pulse
sigma = 4./360./2.355
pulse_amp = 1.
noise_amp = 1.
ph_N = 1000
ph = np.linspace(0, 1, ph_N, endpoint=False)
ph_doub = np.concatenate((ph, ph+1.))
pulse = pulse_amp*np.exp(-0.5*pow((ph-0.5)/sigma, 2))
pulse_doub = np.tile(pulse, 2)
def amp_over_bin(bin_start_phase, binwidth=0.1/p0):
    left = np.searchsorted(ph_doub, bin_start_phase)
    nbins = int(np.round(binwidth * ph_N) + 0.1)
    if np.isscalar(left): left = np.array([left])
    amps = np.zeros(np.shape(left))
    for ii in range(nbins):
        amps += pulse_doub[left+ii]/nbins
        sys.stdout.write("\rProgress: %-5.2f%%" % (100.*float(ii+1)/nbins))
        sys.stdout.flush()
    return amps
"""
"""
fake_data = np.random.normal(scale=noise_amp, size=chd.data.shape)
disp_times = chd.times.repeat(chd.nfreq).reshape(fake_data.shape) - chd.dm_delays(DM, 10000.)
phases = disp_times / p0 % 1.
fake_data += amp_over_bin(phases)

chd_fake = CHIMEdata(data_fname)
chd_fake.data = np.ma.array(fake_data, mask=chd.data.mask)
chd_fake.save_data("fake_dat_lownoise.npy")
"""
"""
chd_fake = CHIMEdata("fake_dat.npy")
chd_fake.data.mask = chd.data.mask
"""
