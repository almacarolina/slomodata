''' A class for processing oceanographic/meteorological
        time series
'''
from __future__ import division
from scipy.signal import detrend
from scipy import stats as stats
import numpy as np
from numpy import linalg as LA
import matplotlib.path as mplPath
from matplotlib.dates import date2num, num2date, datestr2num, datetime
from ocean_filter import butter_bandpass_filter, fir_filter

class time_series():
    def __init__(self,data,time):
        data, time = map(np.asarray, (data, time))
        # Derived information.
        '''time_in_seconds = [(t - time[0]).total_seconds() for t in time]
        dt = np.unique(np.diff(time_in_seconds))
        fs = 1.0 / dt  # Sampling frequency.
        Nyq = fs / 2.0
        self.u = u
        self.time = time
        self.fs = fs
        self.Nyq = Nyq
        self.dt = dt
        self.time_in_seconds = np.asanyarray(time_in_seconds)'''

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


def plot_spectrum(t,u,v,window,ci,nens):
        #compute PSD using simple FF
        t=np.squeeze(np.asarray(t))
        if str(nens)=='None':
                nens=1

        if str(v)=='None':
                whenan=np.isnan(u)
                u[whenan]=0
                data =np.squeeze(np.asarray(detrend(u)))
        else:
                whenan=np.isnan(u)
                u[whenan]=0
                whenan=np.isnan(v)
                v[whenan]=0
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

        dof=nens * 2 * len(window)/sum(window**2)
        alpha1=ci/100
        alpha1=1-alpha1
        alpha1=alpha1/2
        ehi=stats.chi2.ppf(alpha1,dof); elo=stats.chi2.ppf(1-alpha1,dof)
        return freq, PSD, ehi, elo, dof

def coriolis(yc):
        #compute the coriolis and beta parameter
        omega=7.29e-5
        R=6371000
        #positive is counterclockwise, negative is clockwise
        f=2*omega*np.sin(yc*np.pi/180)
        b=2*omega/R*np.cos(yc*np.pi/180)

        return f, b

'''def temp_coverage(U, T):
    """ first check dimensions of hd and hn and U and T, U and T must have nans for empty spaces """
    T = np.squeeze(T)
    ct = np.ones(U.shape[-1])
    # temporal coverage
    for ii in range(0, U.shape[-1]):
        u = np.squeeze(U[ii])
        k = np.where(np.isnan(u)==0)[0].shape[0]
        ct[ii] = (k / (U.shape[-1])) * 100'''


def deg_to_meter(r,zero_mark):
    """r = [lat1,lon1] zero_mark=[lat2,lon2] Convert latitude/longitude distance in degrees
    latitude must be in row 1 and longitude in row 0, all the values are the columns"""
    R = 6371000
    r = np.array(r)
    zero_mark = np.array(zero_mark)
    r =r[:,None]
    zero_mark = zero_mark[:,None]

    rym = (r[1,:] - zero_mark[1,:])*np.pi*R*np.cos(r[1]*np.pi/180)/180
    rxm = (r[0,:] - zero_mark[0,:])/180.*np.pi*R
    rm = np.sqrt((rym/111320)**2+(rxm/111320)**2)

    return rm

'''def make_eof(X):
    "" "computes the eof of a matrix """
    C = np.cov(X)
    L,B = LA.eigh(C)
    L = np.real(L)
    I = L.ravel().argsort()
    L = L[::-1]
    I = I[::-1]
    B = B[:,I]
    bt = B.transpose()
    A = bt.dot(X)

    return A, B, L'''

def interp_nan(data):
    bad_indexes = np.isnan(data)
    good_indexes = np.logical_not(bad_indexes)
    good_data = data[good_indexes]
    interpolated = np.interp(bad_indexes.nonzero()[0], good_indexes.nonzero()[0], good_data)
    data[bad_indexes] = interpolated

    return data

def make_eof(X,a=1):
    """ Computes EOFS OF hfdr ON A TWO DIMENSIONAL MATRIX, when a = 1 (default) number of rows is space and number of columns is time
    when a = 0, rows is time and columns is space """
    if a ==1:
        X = X.T
        print('Rows is space and columns time')
        nt,nx = X.shape
        u, d, vt = np.linalg.svd(X,full_matrices = False) #eigen values are (d ** 2) / nt
        B = vt.T #eigenvectors from SVD (spatial functions)
        A = np.dot(u, np.diag(d)) #temporal functions
        L = (d**2) / nt # eigenvalues normalizados
    else:
        print('Rows is time and Columns is space')
        nx,nt = X.shape
        u, d, vt = np.linalg.svd(X,full_matrices = False) #eigen values are (d ** 2) / nt
        A = vt.T #eigenvectors from SVD (temporal functions)
        B = np.dot(u, np.diag(d)) #spatial functions
        L = (d**2) / nt

    return A, B, L

def calc_ip_f0(lat):
    """ Calculates inertial period and f0 for given latitude in degrees"""
    omega = 7.2921e-5
    f0 = 2 * omega * np.sin( np.deg2rad(lat) )
    ip =  2*np.pi/np.abs(f0)/3600/24
    return ip, f0


'''def masked_interp(t, y):
    """
    gap filling with linear interolation for masked arrays
    loops over 2nd dim and interps masked indices of 1st dim
    """
    yn = y.data.copy().astype(t.dtype)

    for n in range(0, y.shape[1]):
        yn[y[:, n].mask, n] = np.interp(t[y[:, n].mask], t[~y[:, n].mask], y[:, n].compressed())

    return yn'''

def masked_interp_single(t, y):
    """
    gap filling with linear interolation for masked arrays
    loops over 2nd dim and interps masked indices of 1st dim
    """
    yn = y.copy().astype(t.dtype)

    yn[y.mask] = np.interp(t[y.mask], t[~y.mask], y.compressed())

    return yn

def rot_mat(u,v,ang):
    ang = deg2rad(ang)
    ui = np.cos(ang)*u - np.sin(ang)*v
    vi = np.sin(ang)*u + np.cos(ang)*v

    return ui, vi

'''d,v1,u2,v2,window,ci,nens):

            #compute PSD using simple FF
        t=np.squeeze(np.asarray(t));
        if nens==None:
                nens=1

        if v1==None:
                data1 =np.squeeze(np.asarray(u1))
        else:
                data1 =np.squeeze(np.asarray(u1 + 1j * v1)) #data

        if v2==None:
                data2 =np.squeeze(np.asarray(u2))
        else:
                data2 =np.squeeze(np.asarray(u2 + 1j * v2)) #data

        if not len(v1)==len(v2)
                raise ValueError("""Data has different lengths'""")
        whenan=np.isnan(data)
        data[whenan]=0
        if ci==None:
                ci=95

        N=len(data1)
        # to make ensamble averages

        #time interval
        dt = (np.diff(t)).mean()
        df= 1 / (N *dt)

        for i in range(1,15):
            if N<2**i:
                nens=2**i
                break
            else:
                nens=N


        #results will be in days since df is in days
        [Cxy,freq]=cohere(data1, data2, NFFT=nens, df, Fc=0, detrend = mlab.detrend_mean,
         window = mlab.window_hanning, noverlap=0, pad_to=None,
         sides='twosided', scale_by_freq=True)

        return Cxy freq'''

def inside_pol(X,Y,poly):
    '''defines the values that are inside the poligon xi yi, which must be a closed contour from the matrices X Y
    returns both X and Y matrices where values are only nonmasked in the polygon and the indices fi where values are inside'''
    xi = np.reshape(X,(X.shape[0]*X.shape[1]))
    yi = np.reshape(Y,(Y.shape[0]*Y.shape[1]))
    point = np.asarray([xi,yi]).transpose()
    bbbPath = mplPath.Path(poly)
    val = bbbPath.contains_points(point)
    fi = np.where(val == False)[0]
    xi[fi] = np.ma.masked
    yi[fi] = np.ma.masked
    yi = np.ma.masked_equal(yi,0)
    xi = np.ma.masked_equal(xi,0)
    fi = np.where(xi.mask==False)[0]
    return np.reshape(yi,(32,39)), np.reshape(xi,(32,39)), fi

def ps(u,v,dx,dy):

    """ decompose the vector field (u,v) into potential (up,vp)
        and solenoidal (us,vs) fields using 2D FT a la Smith JPO 2008 """

    ix,jx,kx = u.shape
    dl = 1./(ix*dy)
    dk = 1./(jx*dx)
    kNy = 1./(2*dx)
    lNy = 1./(2*dy)
    k = np.arange(-kNy,kNy,dk)
    k = np.fft.fftshift(k)
    l = np.arange(-lNy,lNy,dl)
    l = np.fft.fftshift(l)
    K,L = np.meshgrid(k,l)
    THETA = (np.arctan2(L,K))
    THETA = np.repeat(THETA,kx).reshape(ix,jx,kx)

    u[u.mask==True] = 0
    v[v.mask==True] = 0

    U = np.fft.fft2(u,axes=(0,1))
    V = np.fft.fft2(v,axes=(0,1))

    P = U*np.cos(THETA) + V*np.sin(THETA)
    S = -U*np.sin(THETA) + V*np.cos(THETA)

    # back to physical space
    up = np.real(np.fft.ifft2(P*np.cos(THETA),axes=(0,1)))
    vp = np.real(np.fft.ifft2(P*np.sin(THETA),axes=(0,1)))

    us = np.real(np.fft.ifft2(-S*np.sin(THETA),axes=(0,1)))
    vs = np.real(np.fft.ifft2(S*np.cos(THETA),axes=(0,1)))

    return up,vp,us,vs

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

def band_pass_data_tch(lowi, highi, time, temp):
    # low pass data, lowi must be in hour
    # filter characteristics
    highcut = 1 / (lowi / 24) # to get to days usually 27-40
    lowcut = 1 / (highi / 24)
    wn = [lowcut, highcut]
    dt = (np.diff(time)).mean()
    fs = 1 / dt
    # interpolates masked arrays
    temp_n = temp.data.copy()
    temp = detrend(temp, 1)
    for n in range(0, temp_n.shape[0]):
            temp_fil = np.ma.masked_invalid(temp[n, :])
            if (temp_n[n, temp_fil.mask].size <= temp[n, :].shape[0] * 0.90) & \
               (temp_n[n, temp_fil.mask].size >1):
                    temp_n[n, temp_fil[:].mask] = np.interp(time[temp_fil[:].mask],
                                                   time[~temp_fil[:].mask], temp_fil[:].compressed())
    temp_bp = np.empty_like(temp)
    # high pass
    for n in range(0, temp.shape[0]):
        temp_bp[n, :] = fir_filter(time, temp[n, :], wn, win = 'blackman', ftype = 'band',
                             ntaps = 1001, ax = 0, mode = 'same')
    return temp_bp


def angle_between(p1, p2):
    '''returns the angle in -+ 360 degrees, cw is positive, ccw is negative'''
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi)) # this returns then angle from 0 to 360 cw
    #return np.rad2deg((ang1 - ang2)) # if you change to ang2-ang1 then cw is negative and ccw is positive
