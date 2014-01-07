import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import sys
import scipy.signal as sig

class CHIMEdata:
    """
    CHIME frequency and time data with some pulsar-related methods.

    'data' should either be a single .npy file as saved by the save_data
     method or an ordered list of CHIME hdf5 files.

    Currently assuming that eight feeds took data and summing them as phased
     array *except* (and this is a big missing step!) that currently no phase
     delays are being applied.
    """
    def __init__(self, datafiles):
        # If 'data' is a string, check if it ends with .npy
        if isinstance(datafiles, str):
            if datafiles[-4:] == '.npy':
                self.data, self.fpga_count, self.tres = np.load(datafiles)
            else:
                # Make it a list so it gets caught by the next if statement
                datafiles = [datafiles]

        if not isinstance(datafiles, str):
            dat = h5.File(datafiles[0], mode='r')
            self.tres = dat.attrs['fpga.int_period'][0]
            self.ant_chans = dat.attrs['chan_indices']
            dat.close()
            off_diag = (self.ant_chans['ant_chan_a'] !=\
                self.ant_chans['ant_chan_b']).nonzero()[0]
            chunks = []
            tstamps = []
            for datafile in datafiles:
                print "Loading %s..." % datafile
                dat = h5.File(datafile, mode='r')
                vis = dat['vis'].value['real'] + 1.j*dat['vis'].value['imag']
                vis[:, off_diag, :] *= 2.
                chunks.append(
                    np.abs(vis.sum(axis=1)).astype('float32')
                    #np.abs(vis[:,datachans,:]).sum(axis=1).astype('float32')
                )
                del vis
                tstamps.append(dat['timestamp'].value['fpga_count'])
                dat.close()
                print "Done."
            self.data = np.concatenate(chunks, axis=0)
            self.fpga_count = np.concatenate(tstamps)

            self.data = np.ma.array(self.data,
                                    mask=np.zeros(self.data.shape, dtype=bool))

        self.nfreq = self.data.shape[1]
        self.highfreq = 800.
        self.lowfreq = 400.
        self.freqs = np.linspace(self.highfreq, self.lowfreq, self.nfreq)\
            - (self.highfreq - self.lowfreq)/self.nfreq/2.

        self.nsamp = self.data.shape[0]
        self.times = np.linspace(0., self.nsamp*self.tres, self.nsamp,
                                 endpoint=False)

        # and some placeholders for things that can be calculated if desired
        self.tseries = None
        self.spectrum = None
        self.profile = None
        self.phase_vs_freq = None
        self.phase_vs_time = None

    def show(self, withmask=True):
        plt.figure(figsize=(16, 10))
        if withmask:
            plt.imshow(self.data, aspect='auto', cmap=plt.cm.Greens,
                       interpolation='nearest', origin='lower')
        else:
            plt.imshow(self.data.data, aspect='auto', cmap=plt.cm.Greens,
                       interpolation='nearest', origin='lower')
        plt.xlabel("frequency channel")
        plt.ylabel("time sample")

    def auto_freq_mask(self, threshold=3.):
        """
        Try to mask particular frequency channels that seem to be dominated by
         RFI.
        """
        data = self.data.data
        channels = data.sum(axis=0)
        abs_d2 = np.concatenate((np.zeros(1), np.abs(np.diff(channels, n=2)),\
            np.zeros(1)))
        resamp_d2 =\
            np.array(np.split(abs_d2, self.nfreq/2)).sum(axis=1).repeat(2)
        mask1 = resamp_d2 > threshold*np.median(resamp_d2)
        mask1 += (channels < 1.e-3)
        mask1[0] = True
        mask1[-1] = True
        self.data.mask += np.tile(mask1, data.shape[0]).reshape(data.shape)

    def auto_time_mask(self, threshold=5., dt=0.5):
        """
        Try to mask particular samples channel-by-channel that seem to be RFI
         spikes.  The data are downsampled to time resolution 'dt', and the
         masking is applied to chunks of this size, and one 'dt' to either side
         of any chunk that is identified for masking.

        Time masking should be following by replace_masked_times_with_noise,
         since gaps in time cause problems when dedispersing.
        """
        data = self.data.data
        mask = self.data.mask
        nbins_per_chunk = int(np.rint(dt/self.tres)+0.1)
        splitter = np.arange(0, data.shape[0], nbins_per_chunk)[1:]
        spl = np.array_split(data, splitter)
        dsamp_data = np.array([thing.mean(axis=0) for thing in spl])
        channel_medians = np.median(dsamp_data, axis=0)
        abs_devs_from_med = np.abs(dsamp_data - channel_medians)
        channel_mads = np.median(abs_devs_from_med, axis=0)
        dsamp_time_mask = abs_devs_from_med > threshold*channel_mads
        # Spread out the mask by one additional bin to be safe
        dsamp_time_mask[1:] = dsamp_time_mask[1:] + dsamp_time_mask[:-1]
        dsamp_time_mask[:-1] = dsamp_time_mask[1:] + dsamp_time_mask[:-1]
        tmask = dsamp_time_mask.repeat(nbins_per_chunk, axis=0)[:data.shape[0]]
        self.data.mask += tmask

    def pad_missing_samples(self):
        """
        This checks self.fpga_count for dropped samples and, where present,
        inserts dummy values into the appropriate points in the data and masks
        those values.

        This should almost certainly be followed by a call to
        replace_masked_times_with_noise.
        """
        if self.data.shape[0] > len(self.fpga_count):
            print "The number of samples is greater than the number of "\
                "timestamps.  It looks as though padding was already "\
                "performed on this data."
        else:
            data = self.data.data.copy()
            mask = self.data.mask.copy()

            stepsizes = np.diff(self.fpga_count)
            nsamps_per_samp = stepsizes/np.min(stepsizes)
            insert_before = np.where(nsamps_per_samp > 1)[0] + 1
            # subtract 1 because nsamps_per_samp = 1 means there is no gap
            size_of_gap = nsamps_per_samp[insert_before - 1] - 1

            all_inserts = np.ones((size_of_gap.sum(), data.shape[1]), dtype=bool)
            where_inserts = []
            for ii in range(len(size_of_gap)):
                where_inserts += size_of_gap[ii] * [insert_before[ii]]

            data = np.insert(data,
                             where_inserts,
                             all_inserts.astype(data.dtype),
                             axis=0)

            mask = np.insert(mask,
                             where_inserts,
                             all_inserts,
                             axis=0)

            self.data = np.ma.array(data, mask=mask)
            self.nsamp = self.data.shape[0]
            self.times = np.linspace(0., self.nsamp*self.tres, self.nsamp,
                                     endpoint=False)

    def replace_masked_times_with_noise(self, save_time_mask=True):
        new_mask = np.zeros(self.data.shape, dtype=bool)
        for ii in range(self.nfreq):
            chan_mask = self.data.mask[:,ii].copy()
            n_masked_bins = len(chan_mask.nonzero()[0])
            if self.nsamp - n_masked_bins:
                chan_mean = np.mean(self.data[:,ii])
                chan_std = np.std(self.data[:,ii])
                noise = np.random.normal(chan_mean, chan_std, n_masked_bins)
                self.data.data[chan_mask,ii] = noise
            else:
                new_mask[:,ii] = True
            sys.stdout.write("\rProgress: %-5.2f%%" %\
                (100.*float(ii+1)/self.nfreq))
            sys.stdout.flush()
        if save_time_mask:
            self.time_mask = self.data.mask.copy()
        self.data.mask = new_mask

    def zap_chans(self, first, last=None, samp1=None, samp2=None, unzap=False):
        """
        Mask out a range of frequency channels (specified by channel index).

        If 'last' is not specified, it is assumed to be the same as 'first'
         (ie, only one frequency channel is zapped)
        """
        if samp1 is None: samp1 = 0
        if samp2 is None: samp2 = self.nsamp - 1
        if last is None: last = first
        if not unzap:
            self.data.mask[samp1:(samp2+1), first:(last+1)] = True
        else:
            self.data.mask[samp1:(samp2+1), first:(last+1)] = False

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
        k = (np.rint(k_flt) + 0.1).astype(int)
        tseries = np.zeros(nsamp, dtype=float)
        for ll in range(nfreq):
            if not data.mask[:,ll][0]:
                tseries += np.roll(data[:,ll], k[ll])
        if not no_save:
            self.tseries = TimeSeries(tseries[np.max(k):], self.tres, dm)
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
        data = self.data[start_samp:end_samp, start_chan:end_chan].copy()
        # Dividing out a per-channel running mean (with the default radius of
        # 50 data points here) gets rid of the slow variation of the signal
        # over time and also, unlike subtracting the mean, appears to give the
        # noise the same amplitude across all frequency channels.  However, I
        # am not sure how 'correct' this is conceptually.
        data /= running_mean(data)
        freqs = self.freqs[start_chan:end_chan]
        times = self.times[start_samp:end_samp]
        if kwargs.has_key('f_ref'): f_ref = kwargs['f_ref']
        else: f_ref = freqs[0] 
        profile = np.zeros(nbins, dtype=float)
        ddtimes = times.repeat(len(freqs)).reshape(data.shape)\
            - self.dm_delays(dm, f_ref)[start_chan:end_chan]
        phases = ddtimes / p0 % 1.
        which_bins = (phases * nbins).astype(int)
        for ii in range(nbins):
            vals = data[which_bins == ii]
            profile[ii] += vals.mean()
        if not no_save: self.profile = Profile(profile, p0, dm, nbins)
        return profile

    def calc_phase_vs_freq(self, p0, dm, nbins=32, nsubs=32, dedisp=True,
                           **kwargs):
        """
        Calculate a phase-vs-frequency waterfall plot and store it in
         self.phase_vs_freq
        This can be plotted using self.phase_vs_freq.show()

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
        for ii in range(nsubs):
            start_chan = ii*(self.nfreq/nsubs)
            end_chan = (ii+1)*(self.nfreq/nsubs)
            seg_freqs = self.freqs[start_chan:end_chan]
            if dedisp: f_ref = self.freqs[0]
            else: f_ref = seg_freqs[0]
            row = self.fold_pulsar(p0, dm, nbins,\
                start_samp=start_samp, end_samp=end_samp,
                start_chan=start_chan, end_chan=end_chan, f_ref=f_ref,\
                no_save=True)
            sys.stdout.write("\rProgress: %-5.2f%%" %\
                (100.*float(ii+1)/nsubs))
            sys.stdout.flush()
            waterfall.append(np.tile(row, 2))
        waterfall = np.array(waterfall)
        if not no_save:
            self.phase_vs_freq = PhaseVSFreqPlot(waterfall,\
                low_f, high_f, p0, dm, dedisp)
        return waterfall

    def calc_phase_vs_time(self, p0, dm, nbins=32, nints=32, **kwargs):
        """
        Calculate a phase-vs-time waterfall plot and store it in
         self.phase_vs_time
        This can be plotted using self.phase_vs_time.show()

        kwargs:
          no_save: if this is True, return output but don't store it in object
          (By default, it is False)

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
            waterfall.append(np.tile(row, 2))
            sys.stdout.write("\rProgress: %-5.2f%%" %\
                (100.*float(ii+1)/nints))
            sys.stdout.flush()

        waterfall = np.array(waterfall)
        if not no_save:
            self.phase_vs_time = PhaseVSTimePlot(waterfall,\
                start_t, end_t, p0, dm)
        return waterfall

    def calc_spectrum(self, dm, N=None, whiten=False, **kwargs):
        """
        Not a lot of thought was put into the whitening, and it's almost
         certainly pretty sketchy

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
        Saves the data to a .npy file that can be used in the class
        constructor to reload the saved data.
        """
        np.save(out_fname, np.array([self.data, self.fpga_count, self.tres]))

class PhaseVSFreqPlot:
    def __init__(self, data, low_f, high_f, p0=None, dm=None, dedisp=None):
        self.data = data
        self.low_f = low_f
        self.high_f = high_f
        self.p0 = p0
        self.dm = dm
        self.dedisp = dedisp
    def show(self):
        plt.figure(figsize=(12, 9))
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
        plt.figure(figsize=(12, 9))
        plt.imshow(self.data, aspect='auto', cmap=plt.cm.Greens,
                   interpolation='nearest', origin='lower',
                   extent=(0., 2., self.start_t, self.end_t))
        plt.xlabel("phase")
        plt.ylabel("observing time (s)")

class TimeSeries:
    def __init__(self, data, tres=None, dm=None):
        self.data = data
        self.tres = tres
        self.dm = dm
    def show(self):
        plt.figure(figsize=(12, 9))
        plt.plot(self.data)
    def save_dat(self, fname):
        """
        This saves a data/header pair that can be read in by the pulsar
        search software PRESTO--in particular, its 'prepfold', 'accelsearch',
        'exploredat', and 'realfft' functions
        """
        nsamp = len(self.data)
        if fname.split('.')[-1] == 'dat': fname = fname[:-4]
        if nsamp % 2:
            self.data[1:].astype('float32').tofile(fname + '.dat')
            nsamp -= 1
        else:
            self.data.astype('float32').tofile(fname + '.dat')
        inf_file = open(fname + '.inf', 'w')
        inf_file.write(
            ' Data file name without suffix          =  %s' % fname
            + '\n Telescope used                         =  CHIME'
            + '\n Instrument used                        =  CHIME'
            + '\n Object being observed                  =  NA'
            + '\n J2000 Right Ascension (hh:mm:ss.ssss)  =  00:00:00.0000'
            + '\n J2000 Declination     (dd:mm:ss.ssss)  =  00:00:00.0000'
            + '\n Data observed by                       =  NA'
            + '\n Epoch of observation (MJD)             =  50000.0'
            + '\n Barycentered?           (1=yes, 0=no)  =  1'
            + '\n Number of bins in the time series      =  %d' % nsamp
            + '\n Width of each time series bin (sec)    =  %.2f' % self.tres
            + '\n Any breaks in the data? (1=yes, 0=no)  =  0'
            + '\n Type of observation (EM band)          =  Radio'
            + '\n Data analyzed by                       =  NA'
            + '\n Any additional notes:'
        )
        inf_file.close()

class Spectrum:
    def __init__(self, data, freq_axis, dm=None, whitened=None):
        self.data = data
        self.freq_axis = freq_axis
        self.dm = dm
        self.whitened = whitened
    def show(self):
        plt.figure(figsize=(12, 9))
        plt.plot(self.freq_axis, self.data)
        plt.xlabel("frequency (Hz)")
    def show_power(self):
        plt.figure(figsize=(12, 9))
        plt.plot(self.freq_axis, self.data**2)
        plt.xlabel("frequency (Hz)")

class Profile:
    def __init__(self, data, p0=None, dm=None, nbins=None):
        self.data = data
        self.p0 = p0
        self.dm = dm
        self.nbins = nbins
    def show(self):
        plt.figure(figsize=(12, 9))
        plt.plot(np.linspace(0, 2, 2*len(self.data), endpoint=False),
                 np.tile(self.data, 2))
        plt.xlabel("phase")

def running_mean(arr, radius=50):
    """
    It might be better to use something a little more sophisticated than this,
    but this gets the job done for now.
    """
    n = radius*2+1
    try: mask = arr.mask
    except: pass
    padded = np.concatenate((arr[1:radius+1][::-1], arr,\
        arr[-radius-1:-1][::-1]))
    ret = np.cumsum(padded, axis=0, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    try: return np.ma.array(ret[n-1:]/n, mask=mask)
    except: return ret[n-1:]/n

