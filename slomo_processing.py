''' A class for processing ADCPS and CTDS data in Python 3 '''
from scipy.signal import detrend
from scipy import stats as stats
import scipy as sc
import numpy as np
from numpy import linalg as LA
import matplotlib as mpl
import matplotlib.path as mplPath
from ocean_filter import butter_bandpass_filter, fir_filter
from matplotlib.dates import date2num, num2date, datestr2num, datetime


# Miscellaneous codes
def deg_to_meter(r, zero_mark):
    """Convert latitude/longitude to meters. Perfect sphere approx."""
    R = 6371000   # Earth radius in m
    rxm = (r[0] - zero_mark[0]) * np.pi * R * np.cos(r[1] * np.pi / 180) / 180
    rym = (r[1] - zero_mark[1]) / 180. * np.pi * R
    rm = rxm, rym
    return rm

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def make_time_python(t):
    """ makes matlab time into python time, return values in numbers """
    T = t.copy().astype(t.dtype)
    if np.ma.is_masked(t):
        Ttemp = np.asarray(num2date(t[~t.mask])) - datetime.timedelta(days = 366)
        T[~t.mask] = date2num(Ttemp)
    else:
        T = np.asarray(num2date(t)) - datetime.timedelta(days = 366)
        T = date2num(T)
    return T

def coriolis(yc):
    #compute the coriolis and beta parameter
    omega = 7.29e-5
    R = 6371000
    #positive is counterclockwise, negative is clockwise
    f = 2 * omega * np.sin(yc * np.pi / 180)
    b = 2 * omega / R * np.cos(yc * np.pi / 180)
    return f, b

# Interpolation methods
def colloc_time(t1, t2, dt):
    """ colocate two time series with the same date and dt """
    t = np.arange(max(t1[0], t2[0]), min(t1[-1], t2[-1]) + dt, dt)
    I1 = np.empty(0, dtype=int)
    I2 = np.empty(0, dtype=int)
    I = np.empty(0, dtype=int)
    for i in range(0, len(t)):
        K1 = mpl.mlab.find(np.abs(t1 - t[i]) <= dt / 2)
        if not len(K1) == 0:
            mix1 = abs(t1[K1] - t[i])
            tt1 = min(mix1)
            mix1 = np.squeeze(mix1)
            i1 = np.array(np.where(tt1 == mix1))
            I1 = np.concatenate([I1, K1[i1]], axis=None)
        else:
            I = np.concatenate([I, i], axis=None)
            I1 = np.concatenate([I1, np.nan], axis=None)
        K2 = mpl.mlab.find(np.abs(t2 - t[i]) <= dt / 2)
        if not len(K2) == 0:
            mix2 = abs(t2[K2] - t[i])
            tt2 = min(mix2)
            mix2 = np.squeeze(mix2)
            i2 = np.array(np.where(tt2 == mix2))
            I2 = np.concatenate([I2, K2[i2]], axis=None)
        else:
            I = np.concatenate([I, i], axis=None)
            I2 = np.concatenate([I2, np.nan], axis=None)
    t = np.delete(t, I)
    I1 = np.delete(I1, I)
    I2 = np.delete(I2, I)
    I1 = np.asarray(I1, dtype=int)
    I2 = np.asarray(I2, dtype=int)
    return t, I1, I2


def masked_interp_single(t, y):
    """
    gap filling with linear interolation for masked arrays
    loops over 2nd dim and interps masked indices of 1st dim
    """
    yn = y.copy().astype(t.dtype)
    yn[y.mask] = np.interp(t[y.mask], t[~y.mask], y.compressed())
    return yn


# Filtering methods
def plot_spectrum(t,u,v,window,ci,nens):
        #compute PSD using simple FF
        t=np.squeeze(np.asarray(t))
        if str(nens)=='None':
                nens=1
        # check for NaNs or masked arrays
        if str(v)=='None':
                u = np.ma.masked_invalid(u)
                u[u.mask == True ] = 0
                data =np.squeeze(np.asarray(detrend(u)))
        else:
                u = np.ma.masked_invalid(u)
                v = np.ma.masked_invalid(v)
                u[u.mask == True] = 0
                v[v.mask == True] = 0
                data =np.squeeze(np.asarray(detrend(u) + 1j * detrend(v))) #data

        if ci==None:
                ci=95

        N=len(data)
        # to make ensamble averages
        ne=np.int(N/nens)
        neni=nens*ne
        data=np.reshape(data[0:neni],(ne,nens),'F')
        #time interval
        dt = (np.diff(t)).mean()
        df= 1 / (N *dt)
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman','None']:
                raise ValueError("""Window is on of 'flat', 'hanning', 'hamming',
                         'bartlett', 'blackman'""")

        Nsin=len(data)
        if window != 'None':
                window = eval('np.'+ window + '(Nsin)')
                #create window
        else:
                window=np.asarray(1)
        #REALLY IMPORTANT
        # if it does not have a window  PSD= (dt/N)*abs(foucoff)**2/sum(window**2)
        window.resize((Nsin,) + tuple(np.int8(np.ones(data.ndim - 1))))
        window=np.squeeze(np.array([[window],]*nens)).transpose()
        #to add a window we need normalize the window
        # data = detrend(data.squeeze())
        dataw=np.squeeze(data)*window
        #the window correction only if you are not dividing by sum(window**2) en PSD
        foucoff=  np.fft.fftshift(np.fft.fft(dataw,axis=0),axes=(0,))
        #this is for power spectra and needs correction for window on foucoff
        #PSD= (dt/N)*abs(foucoff)**2
        #this is for power spectral density and needs correction for window in foucoff
        #THIS IS THE CORRECT ONE for pcddddower spectra density but Parseval theorem is sum(PSD * dt**2) * df=var(data)
        #PSD= (1/(N*dt))*abs(foucoff)**2
        #this is cedric and it works with parseval theorem var(data)=sum(PSD)*df is power spectra and it
        PSD= dt*abs(foucoff)**2/sum(window**2)
        # freq=df*np.arange(N/ 2) this is just half of the data with 0 value
        #freq = np.fft.fftfreq(N,dt)
        freq = np.fft.fftshift(np.fft.fftfreq(Nsin, dt))
        #with
        #freq = [0:nr2 , -nr2+1:-1]' / (nr*dt);
        #calculate the 95% confidence limits
        #dof = N/2 * nr / sum(win.^2);
        if nens==1:
                None
        else:
                PSD=np.mean(PSD,axis=1)
        dof = nens * 2 * len(window) / sum(window**2)
        alpha1 = ci/100
        alpha1 = 1 - alpha1
        alpha1 = alpha1/2
        # ehi=stats.chi2.ppf(alpha1,dof); elo=stats.chi2.ppf(1-alpha1,dof)
        ehi, elo = stats.chi2.ppf([alpha1, 1-alpha1], dof[0])
        return freq, PSD, ehi, elo, dof[0]

def make_band_pass_adcp(u, v, time, lowi, highi, tipy):
    # make a band pass, high or low of adcp data of two dimensions where
    # last dimension is depth
    # filter characteristics
    dt = (np.diff(time)).mean()
    fs = 1 / dt
    if highi == 'None':
        wn = 1 / (lowi / 24)
    else:
        highcut = 1 / (lowi / 24) # to get to days usually 27-40
        lowcut = 1 / (highi / 24)
        wn = [lowcut, highcut]
    # interpolates masked arrays
    un = np.ma.masked_array(u, copy=True)
    vn = np.ma.masked_array(v, copy=True)
    for n in range(0, u.shape[-1]):
        ufil = u[:, n]
        vfil = v[:, n]
        if ufil[~ufil.mask].size != ufil.shape[0]:
                un[ufil[:].mask, n] = np.interp(time[ufil[:].mask],
                                               time[~ufil[:].mask], ufil[:].compressed())
                vn[vfil[:].mask, n] = np.interp(time[vfil[:].mask],
                                               time[~vfil[:].mask], vfil[:].compressed())
    # Apply filter
    ub = np.empty_like(un)
    vb = np.empty_like(vn)
    for n in range(0, u.shape[-1]):
        ub[:, n] = fir_filter(time, un[:, n], wn, win = 'blackman', ftype = tipy,
                         ntaps = 1001, ax = 0, mode = 'same')
        vb[:, n] = fir_filter(time, vn[:, n], wn, win = 'blackman', ftype = tipy,
                         ntaps = 1001, ax = 0, mode = 'same')
        ub[u.mask] = np.ma.masked
        vb[u.mask] = np.ma.masked
    return ub, vb

def make_band_pass_1d(u, time, lowi, highi, tipy):
    # make a band pass, high or low of adcp data of two dimensions where
    # last dimension is depth
    # filter characteristics
    dt = (np.diff(time)).mean()
    fs = 1 / dt
    if highi == 'None':
        wn = 1 / (lowi / 24)
    else:
        highcut = 1 / (lowi / 24) # to get to days usually 27-40
        lowcut = 1 / (highi / 24)
        wn = [lowcut, highcut]
    # interpolates masked arrays
    u = np.ma.masked_invalid(u, copy=True)
    un = np.copy(u)
    if u[~u.mask].size != u.shape[0]:
        un[u[:].mask] = np.interp(time[u[:].mask],
                                   time[~u[:].mask], u.compressed())
    #  band pass
    un = fir_filter(time, un, wn, win = 'blackman', ftype = tipy,
                        ntaps = 1001, ax = 0, mode = 'same')
    un = np.ma.masked_values(un,0)
    un[u.mask] = np.ma.masked
    return np.ma.masked_values(un,0)

def low_pass_data_tch(lowi, time, temp, tipy):
    # low pass data, lowi must be in hour
    # filter characteristics
    lowcut = 1 / (lowi / 24) # to get to days usually 27-40
    dt = (np.diff(time)).mean()
    fs = 1 / dt
    # interpolates masked arrays
    temp_n = np.copy(temp[:])
    # temp = detrend(temp, 1)
    for n in range(0, temp_n.shape[0]):
            temp_fil = np.ma.masked_invalid(temp[n, :])
            if (temp_n[n, temp_fil.mask].size <= temp[n, :].shape[0] * 0.90) & \
               (temp_n[n, temp_fil.mask].size >1):
                    temp_n[n, temp_fil[:].mask] = np.interp(time[temp_fil[:].mask],
                                                   time[~temp_fil[:].mask], temp_fil[:].compressed())
    temp_l = np.empty_like(temp)
    # high pass
    for n in range(0, temp.shape[0]):
        temp_l[n, :] = fir_filter(time, temp[n, :], lowcut, win = 'blackman', ftype = tipy,
                             ntaps = 1001, ax = 0, mode = 'same')
    return temp_l


# statistics methods
def corrcoeflevel(N,l):
    """
    compute the null hypothesis value
    of correlation coefficient not
    significantly different from 0
    at the l% confidence level (e.g. 95, 99)
    for N degrees of freedom.
    reference: Emery & Thomson: "Data Analysis Methods in Physical Oceanography"""
    alpha=(100-l)/100
    s=1/np.sqrt(N)
    z=stats.norm.ppf(1-alpha/2)
    Z=z*s
    r=(np.exp(2*Z)-1)/(np.exp(2*Z)+1);

    return r


def deg_to_meter(r, zero_mark):
    """Convert latitude/longitude to meters. Perfect sphere approx."""
    R = 6371000   # Earth radius in m
    rxm = (r[0] - zero_mark[0]) * np.pi * R * np.cos(r[1] * np.pi / 180) / 180
    rym = (r[1] - zero_mark[1]) / 180. * np.pi * R
    rm = rxm, rym
    return rm

def nanxcorrcoef(x,y,l,flag):
    """commputes the (eventually complex) lag correlation coefficient r
    of vectors x and y, ignoring NaN
    for lag, t, from -(N-1) to +(N-1),
    where N is the length of the input vectors.
    n is the number of good points for each lag (rem: if n(t)=0, r(t)=NaN);
    r=nanxcov(x,y,1)./sqrt(nancov(x,x,1).*nancov(y,y,1))
    r0 is the null hypothesis level at the l% confidence (e.g. 95, 99).
    flag=0: remove means before computing r,
    flag=1: do not remove means before computing r.
    reference: Emery & Thomson: "Data Analysis Methods in Physical Oceanography"""

    x=np.asarray(x)
    y=np.asarray(y)

    if not flag in [0,1]:
                 raise ValueError(""" flag must be 0 or 1""")

    if flag==0:
         [cxy,n]=nanxcov(x,y,1)
         [cyx,junk]=nanxcov(y,x,1)
         [cxx,junk]=nanxcov(x,x,1)
         [cyy,junk]=nanxcov(y,y,1)
         tn = np.int(n.size/2)
         r=cxy/np.sqrt(cxx[tn]*cyy[tn])
    else:
         [cxy,n]=nanxcorr(x,y,1)
         [cyx,junk]=nanxcorr(y,x,1)
         [cxx,junk]=nanxcorr(x,x,1)
         [cyy,junk]=nanxcorr(y,y,1)
         tn = np.int(n.size/2)
         r=cxy/np.sqrt(cxx[tn]*cyy[tn])

    if np.remainder(x.size,2)==0:
        t=np.arange(-x.size+1,x.size,1)
    else:
        t=np.arange(-x.size+1,x.size,1)
    N=len(x)
    T=np.median(np.cumsum((cxx*cyy)+(cxy*cyx)))/(cxx[tn]*cyy[tn])
    Nf=N/T
    r0=corrcoeflevel(Nf,l)

    return r, n, r0, t


def nanxcov(x,y,flag):
    """ Computes the lag covariance, c, between vectors x and y for lag t from -(N-1) to +(N-1)
    N is the length of the input vectors
    compute the lag covariance, c, between vectors x and y
    for lag, t, from -(N-1) to +(N-1),
    where N is the length of the input vectors.
    c(t)=E[(x(t'+t)-E[x])*conj(y(t')-E[y])].
    if flag=0, the covariance is normalized by the length of the lagged vectors
    after NaNs have been removed, stored in n  (rem: if n(t)=0, c(t)=NaN);
    if flag=1, the covariance is normalized by n(0)."""
    x=np.transpose(np.asarray(x))
    y=np.transpose(np.asarray(y))
    N=len(x)
    I=np.isnan(x)
    J=np.isnan(y)
    x[I]=0
    y[J]=0
    zx=np.ones(N)
    zx[I]=0
    zy=np.ones(N)
    zy[J]=0
    n=np.correlate(zx,zy,mode='full')

    x=x-np.mean(x)
    y=y-np.mean(y)

    if not flag in [0,1]:
        	raise ValueError("""flag must be 0 or 1""")

    if flag==0:
        c=sc.correlate(x,y,mode='full')/n
    else:
        n0=sc.correlate(zx,zy,mode='valid')
        c=np.correlate(x,y,mode='full')/n0

    fi=np.where(np.asarray(c)==np.inf)
    c[fi]=np.nan
    c= np.ma.masked_array(c, [np.isnan(xf) for xf in c])

    return c, n

def nancorrcoef(x,y,l,flag):
    """same as nanxcorrcoef but for t==0"""

    [r,n,r0,t]=nanxcorrcoef(x,y,l,flag)
    tn=np.where(t==0)
    r=r[tn]

    return r,r0


def ellvar(U, V):
    U = detrend(U)
    V = detrend(V)
    uu = np.nanstd(U) ** 2
    vv = np.nanstd(V) ** 2
    uv = np.cov(U, V)
    Maj = 1 / 2 * (uu + vv + np.sqrt((uu - vv)**2 + 4 * uv**2))
    Min = np.abs(1 / 2 * (uu + vv - np.sqrt((uu - vv)**2 + 4 * uv**2)))
    theta = 1 / 2 * np.arctan2(2 * uv, uu - vv)
    Ener = Maj[0,1] ** 2 + Min[0,1] ** 2
    return Maj, Min, theta, Ener
