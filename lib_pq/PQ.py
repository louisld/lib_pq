from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
from os import path
import os


class PQ():

    def __init__(self, R, N, a):
        self.R = R
        self.N = N
        self.a = a

    def printProgressBar(self, iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
        # Print New Line on Complete
        if iteration == total:
            print()

    @staticmethod
    def kronecker(m, n):
        """
        Fonction de kronecker

        Parameters
        ----------
        m : int
        n : int
        """
        if(m == n):
            return 1
        else:
            return 0

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
        if cache is True:
            if path.exists('cache/h{}.csv'.format(self.N)):
                self.h = np.loadtxt('cache/h{}.csv'.format(self.N), delimiter=',')
                return self.h
        h = np.zeros((self.N, self.N))
        self.printProgressBar(0, self.N, prefix="Calcul de l'hamiltonien :", length=50)
        for n in range(1, self.N+1):
            for m in range(1, self.N+1):
                def g(x):
                    res = np.sin(n*np.pi*np.array(x))
                    res *= np.sin(m*np.pi*np.array(x))
                    res *= self.potentiel(x, **kwargs)
                    return res
                h[n-1, m-1] = n**2*self.kronecker(m, n)
                h[n-1, m-1] += 2*integrate.quad(g, 0, kwargs['a'], full_output=1)[0]
            self.printProgressBar(n, self.N, prefix='Calcul de l\'hamiltonien :', length=50)
        self.h = h
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
        if self.h is None:
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
        plt.plot(n, self.vap, label="$\\epsilon(n)$")
