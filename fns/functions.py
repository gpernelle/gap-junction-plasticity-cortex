from fns.utils import *
import matplotlib


def movingaverage(values, window):
    weigths = np.repeat(1.0, window) / window
    smas = np.convolve(values, weigths, 'valid')
    return smas  # as a numpy array


def chart(list1):
    hour_list = list1
    print(hour_list)
    numbers = [x for x in range(0, 24)]
    labels = [str(x) for x in numbers]
    plt.xticks(numbers, labels)
    plt.xlim(0, 24)
    plt.hist(hour_list)
    plt.show()


def f(w=20, h=3):
    plt.figure(figsize=(w, h), linewidth=0.1)


def readDataFile(path):
    '''
    Read data extracted form graph with GraphClick
    :param path: file path
    :return: x,y
    '''
    x = []
    y = []
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            try:
                x.append(float(row[0]))
                y.append(float(row[1]))
            except:
                pass
    return x, y


def resonanceFS(neuron_model=load_config(), tauv=15):
    '''
    Compute the resonance of the Izh model for FS neurons
    :param tauv:
    :return:
    '''
    print("resonance %d" % tauv)
    T = 10000
    dt = 1
    t = np.arange(0, T, dt)
    F = np.logspace(0.5, 2.3, 50)

    a = neuron_model.I.a
    b = neuron_model.I.b
    c = neuron_model.I.c
    tau_u = neuron_model.I.tau_u
    v_reset = neuron_model.I.v_reset

    res_var = np.empty(len(F), dtype=np.float32)
    for k, f in enumerate(F):
        A = 0.01
        I = A * np.cos(2 * np.pi * f * t / 1000)
        res_v = []
        u = 0
        # izh neuron model for cortical fast spiking neurons (that bursts)
        v = -60
        tau_u = 10
        for i in range(len(t)):
            v += dt / tauv * ((v + a + b) * (v + a) - tau_u * u + 8 * I[i])
            u += dt / tau_u * ((v + a + c) - u)
            if v > 25:
                v = -v_reset
                u += 50
            if i * dt > 1500:
                res_v.append(v / A)
        var = np.var(res_v)
        res_var[k] = var
    return res_var

def fourier(signal, dt=0.1):
    '''
    Return frequency with highest power, and its power
    :param signal: list of 1D numpy array
    :return: frequency, power
    '''
    signal = np.array(signal)
    f_val, p_val = maxPowerFreq(signal[int(signal.shape[0] / 2):], dt / 1000)
    return [f_val, p_val]

def csd(v1, v2, dt=0.1, nperseg=512):
    '''
    Return frequency with highest power, and its power
    :param signal: list of 1D numpy array
    :return: frequency, power
    '''
    return signal.csd(v1, v2, fs=1 / 0.0001, nperseg=nperseg)

def maxPowerFreq(y, dt):
    '''
    :param y: signal
    :param dt: timestep
    :return: the max power of the signal and its associated frequency
    '''
    fs = 1. / dt
    y = y - np.mean(y)
    t = np.arange(0, y.shape[0], 1)
    p1, f1 = psd(y, NFFT=len(t), pad_to=len(t), Fs=fs)
    powerVal = 10 * np.log10(max(p1))
    powerFreq = np.argmax(p1) * np.max(f1) / len(f1)
    return powerFreq, powerVal


def svg2pdf(filename, path='/Users/GP1514/Dropbox/0000_PhD/figures/20160704/'):
    subprocess.check_output(["inkscape", "--file", '%s%s.svg' % (path, filename),
                             '--export-area-drawing', '--without-gui', '--export-pdf',
                             '%s%s.pdf' % (path, filename), '--export-latex'])


def svg2eps(filename, path='/Users/GP1514/Dropbox/0000_PhD/figures/20160704/'):
    subprocess.check_output(["inkscape", '%s%s.svg' % (path, filename),
                             '-E', '%s%s.eps' % (path, filename), '--without-gui',
                             '--export-ignore-filters', '--export-ps-level=3'])


def svg2png(filename, path='/Users/GP1514/Dropbox/0000_PhD/figures/20160704/'):
    subprocess.check_output(["inkscape", '%s%s.svg' % (path, filename),
                             '-e', '%s%s.png' % (path, filename), '--without-gui',
                             '--export-ignore-filters', '--export-png', '-d 300'])


def vmin_vmax(df, kind="burst", v=0):
    data = pd.melt(df, id_vars=['tauv', 'sG'], value_vars=[kind + '1', kind + '2'])
    #     print(data.head())
    if v >= 0:
        vmin, vmax = np.percentile(data['value'], v), np.percentile(data['value'], 100)
    else:
        vmin = None
        vmax = None
    # print(vmin, vmax)
    return vmin, vmax


def facet_heatmap(data, df=None, v=0, vmin=None, vmax=None, index='tauv', columns='sG', **kws):
    kind = data['variable'].get_values()[0][:-1]
    if vmin == None:
        vmin, _ = vmin_vmax(df, kind=kind, v=v)
    if vmax == None:
        _, vmax = vmin_vmax(df, kind=kind, v=v)
    data = data.pivot(index=index, columns=columns, values='value')
    im = sns.heatmap(data, yticklabels=10, xticklabels=10, vmin=vmin, vmax=vmax, **kws)  # <-- Pass kwargs to heatmap
    im.invert_yaxis()


def plotGridHeatmap(df, col_wrap=2, cols=['burst1', 'spike1', 'burst2', 'spike2'], v=-1, vmin=None, vmax=None, **kws):
    data = pd.melt(df, id_vars=['tauv', 'sG'], value_vars=cols)
    #     print(data.head())
    with sns.plotting_context(font_scale=5.5):
        g = sns.FacetGrid(data, col="variable", col_wrap=col_wrap, size=3, aspect=1)

    cbar_ax = g.fig.add_axes([.92, .3, .02, .4])  # <-- Create a colorbar axes
    g = g.map_dataframe(facet_heatmap, v=v, df=df, vmin=vmin, vmax=vmax,
                        cbar_ax=cbar_ax, **kws)  # <-- Specify the colorbar axes and limits
    g.set_titles(col_template="{col_name}", fontweight='bold', fontsize=18)
    g.fig.subplots_adjust(right=.9)  # <-- Add space so the colorbar doesn't overlap the plot
    return g


pd.options.mode.chained_assignment = None

def plotHeatmap(df, col="cor1", title='', cmap=None, y='tauv', x='sG', xres=10, yres=10, **kws):
    plt.figure()
    '''
    plot heatmap using seaborn library
    '''
    burst = df[[y, x, col]]
    burst.loc[:, (col)] = burst[col].astype(float)
    burst.loc[:, (y)] = burst[y].astype(float)
    burst.loc[:, (x)] = burst[x].astype(float)
    c = burst.pivot(y, x, col)

    im = sns.heatmap(c, yticklabels=yres, xticklabels=xres, cmap=cmap, **kws)
    im.invert_yaxis()

    if not title:
        title = col
    plt.title(title)
    return im


def generateInput(seed, T, n=30):
    '''
    Generate a periodic signal
    :param seed:
    :param T:
    :param n: 
    :return:
    '''
    dt = 0.00025
    np.random.seed(seed)
    x = np.linspace(0.0, dt * T, T)
    y = np.zeros(len(x))
    for i in range(5, 300, n):
        y += np.random.rand() * np.sin(i * 2.0 * np.pi * x)
    return y / np.max(y)


def generateInput2(seed, T, n=None, tau=10):
    '''
    Generate colored noise
    :param seed:
    :param T:
    :param n:
    :return:
    '''
    scaling = 1 / (1 / (2 * 2 / 0.25)) ** 0.5 * 70
    dt = 0.1
    np.random.seed(seed)
    x = np.linspace(0.0, dt * T, T)
    signal = np.zeros(len(x))
    iBack = 0
    for i in range(len(x)):
        iBack = iBack + dt / tau * (-iBack) + (np.random.rand() - 0.5)
        iEff = iBack * scaling
        signal[i] = iEff
    signal -= np.min(signal)
    return signal / np.max(signal)


def interpolate(morepoints, lesspoints):
    '''
    return the linear interpolation of lesspoints to have the same number of points are morepoints
    '''

    x = np.arange(len(lesspoints))
    y = lesspoints

    xvals = np.linspace(0, len(lesspoints), len(morepoints))
    yinterp = np.interp(xvals, x, y)

    return xvals, yinterp


def decode(inp, ifr, taufilt=3, avgper=1, spikeThresh=5):
    ### filtering IFR
    sp2 = 0
    sp2_ = []
    dt = 0.1

    for t in range(len(ifr)):
        sp2 = sp2 + dt / taufilt * (-sp2 + ifr[t])
        sp2_.append(sp2)

    # threshold to detect bursts
    spikes = np.array(sp2_) > spikeThresh

    # get position of rising and falling time
    isi = np.where(np.diff(spikes * 1) != 0)[0]

    # period = difference between rising times
    period = np.diff(isi[::2])

    try:
        decoded = 1 / movingaverage(period, avgper)

        xdec, ydec = interpolate(inp, decoded)
        corr_predict = np.corrcoef(inp, ydec)[0, 1]
    except:
        decoded, xdec, ydec = np.zeros(len(inp)), np.zeros(len(inp)), np.zeros(len(inp))
        corr_predict = 0

    return spikes, xdec, ydec, corr_predict


def getInpPer(inp, spikes):
    # delta = time between spikes
    deltas = (spikes[1:] * 1 - spikes[:-1] * 1)
    # rising slopes
    pos = np.array(np.where(deltas == 1)).squeeze()

    pos = np.array(pos)
    # Time of each period
    # Duration of each period: Rising(t+1) - Rising(t)
    per = pos[1:] - pos[:-1]
    res2_ = []
    for i in range(len(per) - 1):
        res2_.append([per[i], inp[pos[i]]])
    res2_ = np.array(res2_)
    return res2_


def generateConstantInput(totalTime=4000, meanTime=2000, var=100, N=1000):
    '''
    Generate a matrix of dim N x Time with 1 in (i,j) if the input of Ni at Tj is on.
    :param totalTime: simulation duration
    :param meanTime: mean time of input arrival
    :param var: controls the jittering (0 for no jittering)
    :param N: number of neurons
    :return: input matrix
    '''
    jitter = np.random.normal(meanTime, var, N)
    inputSig = np.empty((N, totalTime))
    for i in range(totalTime):
        inputSig[:, i] = jitter < i
    return inputSig


def plotFFT(y, T):
    dt = 0.00025
    yf = fft(y)
    xf = np.linspace(0.0, 1.0 / (2.0 * dt), T / 2)
    plt.figure()
    plt.plot(xf, 2.0 / dt * np.abs(yf[0:T / 2]))
    plt.xlim([0, 300])


def facet_heatmap2(data, col='cor1', cols=['cor1', 'cor2', 'corChange'], **kws):
    data = data[data['variable'] == col]
    data = data.pivot(index='tauv', columns='sG', values='value')
    im = sns.heatmap(data, **kws)
    im.invert_yaxis()


def plotGrid(df, col, title='', cols=['cor1', 'cor2', 'corChange'], **kws):
    data = pd.melt(df, id_vars=['tauv', 'sG', 'T', 'both'], value_vars=cols)

    with sns.plotting_context(font_scale=5.5):
        g = sns.FacetGrid(data, col="both", row="T")
    g = g.map_dataframe(facet_heatmap2, col=col, cols=cols, **kws)

    plt.subplots_adjust(top=0.9)
    g.fig.suptitle(title, fontsize='16')
    g.savefig(DIRECTORY + 'cor-plot_%s.png' % col)


def find(arr, v):
    idx = (np.abs(arr - v)).argmin()
    return idx


def convertRaster(r):
    T = r.shape[1]
    x, y = [], []
    for i in range(T):
        yi = np.ravel(np.where(r[:, i] == 1)).tolist()
        y.append(yi)
        x.append(np.ones(len(yi)) * i)
    x = np.concatenate(x)
    y = np.concatenate(y)
    return x, y


def plotRaster(r, w=6, h=3):
    a = w
    b = h
    x, y = convertRaster(r.transpose())
    fig = plt.figure(figsize=(a, b))
    ax = fig.add_subplot(111)
    ax.plot(x, y, '.', color='grey', alpha=0.1)


def norm(x):
    return x / np.max(x)


def plotGrid(df, col, title='', cols=['cor1', 'cor2', 'corChange'], **kws):
    data = pd.melt(df, id_vars=['tauv', 'sG', 'end', 'net'], value_vars=cols)

    with sns.plotting_context(font_scale=5.5):
        g = sns.FacetGrid(data, col="end", row="net")
    cbar_ax = g.fig.add_axes([.85, .3, .02, .4])
    g = g.map_dataframe(facet_heatmap2, col=col, cols=cols, cbar_ax=cbar_ax, **kws)
    for ax in g.axes.flat:
        ax.set_title("")
        ax.set_yticklabels([int(x) for x in tauvlist[::-1]])
        ax.set_xticklabels([int(x) for x in sGlist])
    g.fig.subplots_adjust(top=0.9, right=.8)
    g.fig.suptitle(title, fontsize='16')


def plotGrid2(df, col, title='', cols=['cor1', 'cor2', 'corChange'], **kws):
    data = pd.melt(df, id_vars=['tauv', 'sG', 'end'], value_vars=cols)

    with sns.plotting_context(font_scale=5.5):
        g = sns.FacetGrid(data, col="end")
    cbar_ax = g.fig.add_axes([.85, .3, .01, .4])
    g = g.map_dataframe(facet_heatmap2, col=col, cols=cols, cbar_ax=cbar_ax, **kws)
    for ax in g.axes.flat:
        ax.set_title("")
        ax.set_yticklabels([int(x) for x in tauvlist[::-1]])
        ax.set_xticklabels([int(x) for x in sGlist])
    ax.set_ylabel(r'$\tau_v$')
    plt.subplots_adjust(top=0.9, right=0.8)
    g.fig.suptitle(title, fontsize='16')


def mutualInformation(X, Y, bins):
    c_XY = np.histogram2d(X, Y, bins)[0]
    c_X = np.histogram(X, bins)[0]
    c_Y = np.histogram(Y, bins)[0]

    H_X = shan_entropy(c_X)
    H_Y = shan_entropy(c_Y)
    H_XY = shan_entropy(c_XY)

    MI = H_X + H_Y - H_XY
    return MI


def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized * np.log2(c_normalized))
    return H


def getCustomSymbol1(path_index=1):
    if path_index == 1:  # upper triangle
        verts = [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 0.0), ]
    else:  # lower triangle
        verts = [
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (0.0, 0.0), ]
    codes = [matplotlib.path.Path.MOVETO,
             matplotlib.path.Path.LINETO,
             matplotlib.path.Path.LINETO,
             matplotlib.path.Path.CLOSEPOLY,
             ]
    pathCS1 = matplotlib.path.Path(verts, codes)
    return pathCS1, verts


def plot_mat(matrix=np.random.rand(10, 10), path_index=1, alpha=1.0, vmin=0., vmax=1., cmap='BuPu', x_step=1, y_step=1):
    nx, ny = matrix.shape
    X, Y, values = zip(*[(i, j, matrix[i, j]) for i in range(nx) for j in range(ny)])
    marker, verts = getCustomSymbol1(path_index=path_index)
    ax = plt.subplot(111)
    ax.set_xticks(np.arange(nx))
    ax.set_xticklabels(np.arange(nx) * x_step)
    ax.set_xlabel('Number of shared GJs')

    ax.set_yticks(np.arange(ny))
    ax.set_yticklabels(np.arange(ny) * y_step)
    ax.set_ylabel(r'$\Delta f_{res}$')

    im = ax.scatter(X, Y, s=4000,
                    marker=marker,
                    c=values,
                    cmap=cmap,
                    alpha=alpha,
                    vmin=vmin, vmax=vmax)
    return im
