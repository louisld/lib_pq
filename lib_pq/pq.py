from scipy import integrate, optimize
import numpy as np
import matplotlib.pyplot as plt
from os import path
import os

from .utils import printProgressBar, kronecker


class PQ():

    def __init__(self, R, N, a):
        self.R = R
        self.N = N
        self.a = a

    def calcHamiltonien(self, cache=True, **kwargs):
        """
        Retourne la matrice de l'hamiltonien.

        Parameters
        ----------
        cache : boolean
            La fonction doit-elle utiliser le cache ?
        **kwargs :
            Paramètres du potentiel
        """
        if self.potentiel is None:
            raise ValueError("La fonction potentiel n'a pas été définie")
        # Lecture du cache
        if cache is True:
            if path.exists('cache/h{}.csv'.format(self.N)):
                self.h = np.loadtxt('cache/h{}.csv'.format(self.N), delimiter=',')
                return self.h

        h = np.zeros((self.N, self.N))
        printProgressBar(0, self.N, prefix="Calcul de l'hamiltonien :", length=50)
        for n in range(1, self.N+1):
            for m in range(1, self.N+1):
                def g(x):
                    res = np.sin(n*np.pi*np.array(x))
                    res *= np.sin(m*np.pi*np.array(x))
                    res *= self.potentiel(x, **kwargs)
                    return res
                h[n-1, m-1] = n**2*kronecker(m, n)
                h[n-1, m-1] += 2*integrate.quad(g, 0, kwargs['a'], full_output=1)[0]
            printProgressBar(n, self.N, prefix='Calcul de l\'hamiltonien :', length=50)
        self.h = h

        # Écriture du cache
        if cache is True:
            csvdata = np.asarray(self.h)
            os.makedirs('./cache', exist_ok=True)
            np.savetxt('cache/h{}.csv'.format(self.N), csvdata, delimiter=",")

        return h

    def calcElementsPropres(self):
        """
        Calcule les éléments propres de la matrice M
        """
        ep = np.linalg.eig(self.h)
        vap = ep[0]
        vep = np.transpose(ep[1])
        index = vap.argsort()
        self.vap, self.vep = vap[index], vep[index]
        return self.vap, self.vep

    def phi_puit_infini(self, n, x):
        """
        Retourne les fonctions d'ondes du puit de
        potentiel infini.

        Parameters
        ----------
        n :
            Niveau d'énergie
        x:
            Position dans l'espace
        """
        return np.sqrt(2/self.a)*np.sin((n+1)*np.pi*x)

    def proj_puit_infini(self, x, n):
        """
        Projette les vecteurs propres en représentation
        {|x>} dans le puit de potentiel infini.

        Parameters :
        ------------
        x : float
            Position dans l'espace
        n : int
            Niveau d'énergie
        """
        psi = 0
        for j in range(self.N):
            psi += self.vep[n][j]*self.phi_puit_infini(j, x)
        return psi

    def plot_potentiel_puit_infini(self, ax):
        """
        Trace la courbe du potentiel dans un puit infini.

        Parameters :
        ------------
        ax : matplotlib.axes.Axes
            Axe matplotlib sur le quel tracer le graphe
        """
        ax.vlines(0, 0, 5000, color="black")
        ax.hlines(0, 0, self.a, color="black")
        ax.vlines(self.a, 0, 5000, color="black")
        x = np.linspace(0, self.a, num=1000)
        ax.plot(x, [self.potentiel(i, self.a) for i in x])

    def plot_energie_level(self, n):
        """
        Trace le graphe de l'énergie potentiel à côté
        de celui de la fonction d'onde donné par
        plot_potentiel_puit_infini

        Parameters:
        -----------
        n : int
            Niveau d'énergie
        """
        self.calcHamiltonien(a=self.a)
        self.calcElementsPropres()
        fig, axs = plt.subplots(1, 2)
        self.plot_potentiel_puit_infini(axs[0])
        axs[0].hlines(self.vap[n], 0, self.a, color="red")
        x = np.linspace(0, 1, num=1000)
        axs[1].plot(x, self.proj_puit_infini(x, n))

        axs[0].set_ylabel("Énergie")
        axs[1].set_ylabel("$\\psi_{{{}}}$".format(n))
        axs[1].set_xlabel("$x$")
        fig.legend()

    def plot_energie(self):
        self.calcHamiltonien(a=self.a)
        self.calcElementsPropres()
        n = range(1, self.N+1)
        plt.plot(n, self.vap, label="$\\epsilon(n)$", linestyle="", marker="+")
        plt.xlabel("n")
        plt.ylabel("Énergie")

    def plot_fit_energie(self):
        zone1 = np.array(range(0, 17))
        zone2 = np.array(range(18, 33))
        zone3 = np.array(range(33, 49))

        def f(x, a, b, c):
            return a*x**2+b*x+c

        fit1, _ = optimize.curve_fit(f, zone1, self.vap[0:17])
        print("zone 1 : {}".format(fit1))
        fit2, _ = optimize.curve_fit(f, zone2, self.vap[18:33])
        print("zone 2 : {}".format(fit2))
        fit3, _ = optimize.curve_fit(f, zone3, self.vap[33:49])
        print("zone 3 : {}".format(fit3))
        plt.plot(zone1 + 1, f(zone1, fit1[0], fit1[1], fit1[2]), label="Fit 1")
        plt.plot(zone2 + 1, f(zone2, fit2[0], fit2[1], fit2[2]), label="Fit 2")
        plt.plot(zone3 + 1, f(zone3, fit3[0], fit3[1], fit3[2]), label="Fit 3")
        plt.legend()
