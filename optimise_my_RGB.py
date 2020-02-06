import emcee 
import corner 
import pickle
import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from IPython.display import display, Math
from multiprocessing import Pool


def plot_periodogram(x, y, y_s, name):
    
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(x, y, lw=0.5, color="grey")
    ax.plot(x, y_s, lw=2, color="red")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r'Power $\frac{1}{\mu Hz}$', fontsize=20)
    ax.set_xlabel(r'Frequency $\mu Hz$', fontsize=20)
    ax.set_xlim(0.5, 5000)
    ax.set_ylim(1, 10**5)
    fig.savefig(f"{name}.png")
    plt.close()

def plot_solution(x, y, theta, name, star, score):
    
    wn, A, w1, w2, w3, b1, b2, b3, vmax, amp, sigma, log_f = theta

    z = (2*(2)**(1/2))/(np.pi)
    r = (np.sinc((x)/(2*283.212)))**2

    B1 = A*(z*(w1)/(1+((x/b1)**4)))
    B2 = A*(z*(w2)/(1+((x/b2)**4)))
    B3 = A*(z*(w3)/(1+((x/b3)**4)))

    G = amp*np.exp(-(x - vmax)**2/(2*sigma**2))
    bkg = wn + r*B1 + r*B2 + r*B3
    data = r*G
    y_prime = bkg + data

    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(x, y, lw=0.5, color="grey")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r'Power $\frac{1}{\mu Hz}$', fontsize=20)
    ax.set_xlabel(r'Frequency $\mu Hz$', fontsize=20)
    ax.set_xlim(0.5, 5000)
    ax.set_ylim(1, 15**5)
    ax.hlines(wn, xmin=0, xmax=5000, lw=1, color="green", zorder=1000, label="Flat Noise Level")
    ax.plot(x, r*B1, lw=1, color="orange", zorder=1000, label="B")
    ax.plot(x, r*B2, lw=1, color="orange", zorder=1000)
    ax.plot(x, r*B3, lw=1, color="orange", zorder=1000)
    ax.plot(x, r*G, lw=1, color="blue", zorder=1000, label="G")
    ax.plot(x, y_prime, lw=2, color="red", zorder=1000, label="Total Model")
    ax.set_title(f"{star}  {score}")
    ax.legend()
    fig.savefig(f"{name}.png")
    plt.close()

def get_raw_data(star, miss):

    search_result = lk.search_lightcurvefile(star, mission=miss)
    print("found")
    files = search_result.download_all()
    print("downloaded")
    lc = files.PDCSAP_FLUX.stitch()
    lc = lc.remove_outliers().remove_nans()
    pg = lc.to_periodogram(method='lombscargle', normalization='psd')
    pg_nu_guess = lc.to_periodogram(method='lombscargle', normalization='psd',minimum_frequency = 20, maximum_frequency = 300)

    pg_smooth_2 = pg_nu_guess.smooth(method='boxkernel', filter_width=10)

    p = pg.power.value*10**12
    f = pg.frequency.value

    p2 = pg_smooth_2.power.value*10**12
    f2 = pg_smooth_2.frequency.value

    return f, p, f2, p2

def model(theta, x):
    
    wn, A, w1, w2, w3, b1, b2, b3, vmax, amp, sigma, log_f = theta

    z = (2*(2)**(1/2))/(np.pi)
    r = (np.sinc((x)/(2*283.212)))**2

    B1 = A*(z*(w1)/(1+((x/b1)**4)))
    B2 = A*(z*(w2)/(1+((x/b2)**4)))
    B3 = A*(z*(w3)/(1+((x/b3)**4)))

    G = amp*np.exp(-(x - vmax)**2/(2*sigma**2))
    bkg = wn + r*B1 + r*B2 + r*B3
    data = r*G
    y_prime = bkg + data
    return y_prime

def lnlike(theta, x, y):
    
    wn, A, w1, w2, w3, b1, b2, b3, vmax, amp, sigma, log_f = theta

    y_prime = model(theta, x)
    sigma2 = y_prime**2 * np.exp(2*log_f)
    ll = -0.5*np.sum(((y - y_prime)**2 / sigma2) + np.log(sigma2))
    return ll


def conditions(v):
    wn, A, w1, w2, w3, b1, b2, b3, vmax, amp, sigma, log_f = v

    if (w1 > w2) and (w2 > w3) and (b3 > b2) and (b2 > b1) and (w1 < 5) and (w3 > 0.01):
        m=0
    else:
        m=1


    return m

def opt(f, p, init, bnds):
    nll = lambda *args: -lnlike(*args)
    cons = {'type':'ineq', 'fun': conditions}
    soln = minimize(nll, init, args=(f, p), bounds = bnds, constraints=cons, options={'disp': True})
    return soln.x

def estimate_w(f, p):
    m = 8
    return m

def estimate_A(f, p):
    m = np.mean(p[:200])*1.1

    return m, 0.8, 0.15, 0.05

def estimate_B(f, p):
    return 10, 50, 200

def estimate_sigma(numax):
	return 1.5*0.267*numax**0.768

def vmax_guess(f,p, one_value):
    num = one_value

    cvs = np.zeros(num)
    nus = np.zeros(num)
    entries = len(f)/num
    entries = np.int_(entries)
    for i in range(num):
        mean_nu = np.mean(f[i*entries:(i+1)*entries])
        nus[i] = mean_nu
        values = p[i*entries: (i+1)*entries]
        curr_mean = np.mean(values)
        curr_std = np.std(values)
        curr_cv = curr_std/curr_mean
        cvs[i] = curr_cv
    highest = [x[0] for x in sorted(enumerate(cvs),key = lambda x: x[1])[-5 :]]
    vmax_guess = np.mean(nus[highest])

    return vmax_guess


def estimate_amp(f, p, numax):
    idx = np.where((numax - 0.1*numax < f) & (f < numax + 0.1*numax))
    amp = 1.5*np.mean(p[idx])
    return amp


def prepare_guess(f, p, f_guess, p_guess, n):

    W = estimate_w(f, p)
    A, w1, w2, w3 = estimate_A(f, p)
    b1, b2, b3 = estimate_B(f, p)
    numax = vmax_guess(f_guess, p_guess, n)
    sigma = estimate_sigma(numax)
    amp = estimate_amp(f_guess, p_guess, numax)

    return W, A, w1, w2, w3, b1, b2, b3, numax, amp, sigma




#stars = ("HIP103836", "HIP105854", "HIP114842", "HIP109584", "HIP12871", "HIP113801", "HIP4293", "HIP112612", "HIP655", "HIP113148", "HIP103836", "HIP105854", "HIP106566", "HIP117075", "HIP1766", "HIP114775", "HIP117659", )

star = "HD48381"


try:
    d = open(f"{star}.pkl", "rb")
    data = pickle.load(d)
    print("opening pickled data")
    f1 = data[0]
    p1 = data[1]
    f2 = data[2]
    p2 = data[3]

except:
    print("not pickled, searching for data")
    f1, p1, f2, p2 = get_raw_data(star, "TESS")
    print("got star")
    data = (f1, p1, f2, p2)
    with open(f"{star}.pkl",'wb') as f:
        pickle.dump(data, f)




scores = []
results = []
nums = (25, 50, 75, 100, 125,  150, 175, 200)
for i in nums:

    W, A, w1, w2, w3, b1, b2, b3, numax, amp, sigma = prepare_guess(f1, p1, f2, p2, i)

    bnds = ((2, 30), (A-0.2*A, A+0.2*A), (0, 5), (0, 2), (0, 2), (0, 200), (0, 300), (0, 500), (10, 350), (800, 10000), (sigma*0.5, sigma*1.5), (5e-10, 0.5))
    initial = np.array([W, A, w1, w2, w3, b1, b2, b3, numax, amp, sigma, 0.2])
    results_opt = opt(f1, p1, initial, bnds)



    y_prime = model(results_opt, f1)
    sigma2 = y_prime**2 * np.exp(2*results_opt[11])
    ll = -0.5*np.sum(((p1 - y_prime)**2 / sigma2) + np.log(sigma2))
    scores.append([ll])
    results.append(results_opt)
    plot_solution(f1, p1, results_opt, star+ '_' + str(i), star, ll)

print(star)
print(np.max(scores))
print(results[np.argmax(scores)])
print(nums[np.argmax(scores)])