import emcee 
import corner 
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

def plot_solution(x, y, theta, name):
    
    wn, A, w2, w3, p2, p3, vmax, amp, sigma, log_f = theta

    w1=1
    p1=0

    z = (2*(2)**(1/2))/(np.pi)
    r = (np.sinc((x)/(2*283.212)))**2

    B1 = A*(z*(w1)/(1+((x)**4)/(A*(1/w1)*(10**p1))))
    B2 = A*(z*(w2)/(1+((x)**4)/(A*(1/w2)*(10**p2))))
    B3 = A*(z*(w3)/(1+((x)**4)/(A*(1/w3)*(10**p3))))

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
    ax.legend()
    fig.savefig(f"{name}.png")
    plt.close()

def get_raw_data(star, miss):

    search_result = lk.search_lightcurvefile(star, mission=miss)
    files = search_result.download_all()
    lc = files.PDCSAP_FLUX.stitch()
    lc = lc.remove_outliers().remove_nans()
    pg = lc.to_periodogram(method='lombscargle', normalization='psd')
    pg_nu_guess = lc.to_periodogram(method='lombscargle', normalization='psd',minimum_frequency = 1, maximum_frequency = 300)

    pg_smooth_2 = pg_nu_guess.smooth(method='boxkernel', filter_width=10)

    p = pg.power.value*10**12
    f = pg.frequency.value

    p2 = pg_smooth_2.power.value*10**12
    f2 = pg_smooth_2.frequency.value

    return pg, f, p, f2, p2

def model(theta, x):
    
    wn, A, w2, w3, p2, p3, vmax, amp, sigma, log_f = theta

    w1=1
    p1=0

    z = (2*(2)**(1/2))/(np.pi)
    r = (np.sinc((x)/(2*283.212)))**2

    B1 = A*(z*(w1)/(1+((x)**4)/(A*(1/w1)*(10**p1))))
    B2 = A*(z*(w2)/(1+((x)**4)/(A*(1/w2)*(10**p2))))
    B3 = A*(z*(w3)/(1+((x)**4)/(A*(1/w3)*(10**p3))))

    G = amp*np.exp(-(x - vmax)**2/(2*sigma**2))
    bkg = wn + r*B1 + r*B2 + r*B3
    data = r*G
    y_prime = bkg + data
    return y_prime

def lnlike(theta, x, y):
    
    wn, A, w2, w3, p2, p3, vmax, amp, sigma, log_f = theta

    y_prime = model(theta, x)
    sigma2 = y_prime**2 * np.exp(2*log_f)
    ll = -0.5*np.sum(((y - y_prime)**2 / sigma2) + np.log(sigma2))
    return ll

def opt(f, p, init, bnds):
    nll = lambda *args: -lnlike(*args)
    soln = minimize(nll, init, args=(f, p), bounds=bnds, options={'disp': True})
    return soln.x

def estimate_w(f, p):
    m = 8
    return m

def estimate_A(f, p):
    m = np.mean(p[:200]) + 0.5*np.mean(p[:200])

    return m, 1, 1-0.8, 1-0.99

def estimate_B(f, p):
    return 0, 2, 4

def estimate_sigma():
	return 1.5*0.267*30**0.768

def prepare_guess(f, p, f_guess, p_guess):

    W = estimate_w(f, p)
    A, w1, w2, w3 = estimate_A(f, p)
    p1, p2, p3 = estimate_B(f, p)
    numax = vmax_guess(f_guess,p_guess)
    sigma = estimate_sigma()
    

    return W, A, w2, w3, p2, p3, numax, sigma



def vmax_guess(f,p):
	num = 45
	cvs = np.zeros(num)
	nus = np.zeros(num)
	entries = len(f)/num
	entries = np.int_(entries)
	print(f,p,entries)
	for i in range(num):
	    print(i)
	    mean_nu = np.mean(f[i*entries:(i+1)*entries])
	    nus[i] = mean_nu
	    values = p[i*entries: (i+1)*entries]
	    curr_mean = np.mean(values)
	    curr_std = np.std(values)
	    curr_cv = curr_std/curr_mean
	    cvs[i] = curr_cv

	highest = [x[0] for x in sorted(enumerate(cvs),key = lambda x: x[1])[-1 :]]
	vmax_guess = np.mean(nus[highest])
	return vmax_guess


star = "HD203949"

pg1, f1, p1, f2, p2 = get_raw_data(star, "tess")

pg_smooth1 = pg1.smooth(method='boxkernel', filter_width=10.)
p_s1 = pg_smooth1.power.value*10**12

plot_periodogram(f1, p1, p_s1, star)


W, A, w2, w3, p2, p3, numax, sigma = prepare_guess(f1, p_s1, f2, p2)
print(numax)
print(W, A, w2, w3, p2, p3, 230, 100, 100, 0.2)
bnds = ((2, 30), (A-0.2*A, A+0.2*A), (0.1, 0.67), (0.01, 0.5), (1e-6, 5), (3, 10), (1, 300), (1000, 4000), (sigma -0.2*sigma, sigma + 0.2*sigma), (5e-10, 0.5))
initial = np.array([W, A, w2, w3, p2, p3, numax, 1000, sigma, 0.2])
results_opt = opt(f1, p_s1, initial, bnds)
print(results_opt)
plot_solution(f1, p1, results_opt, 'opt_' + star)



print(results_opt[6],results_opt[8])



y_prime = model(initial, f1)
sigma2 = y_prime**2 * np.exp(2*0.2)
ll = -0.5*np.sum(((p1 - y_prime)**2 / sigma2) + np.log(sigma2))
#print(ll)