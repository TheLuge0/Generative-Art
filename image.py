from PIL import Image
from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import math
import colorsys


# Constantes
WIDTH = 1080
HEIGHT = 1920
N = 10000000

# Coloration disponible via cmap
palettes = ['CMRmap', 'afmhot', 'bone', 'copper', 'cubehelix', 'gist_earth' ,'gist_grey', 'gist_heat', 'gist_stern', 'gnuplot', 'gnuplot2', 'hot', 'inferno', 'magma', 'nipy_spectral']


def init_cons(mode: int) -> tuple:
    """Fonction pour initialiser les constantes des équations

    Args:
        mode (int): Type d'équations

    Returns:
        tuple: Tuple des quatre constantes nécessaires
    """
    if mode == 1:
        a, b, c, d = np.random.uniform(-3.0, 3.0, size=4)
    else:
        a, b, c, d = np.random.uniform(-2*np.pi, 2*np.pi, size=4)

    return a, b, c, d


@jit(nopython=True, fastmath=True)
def init(mode: int, a: int, b: int, c: int, d: int) -> tuple:
    """Fonction qui calcul les valeurs min/max a partir des quatre constantes

    Args:
        mode (int): Type d'équations
        a (int): Constante
        b (int): Constante
        c (int): Constante
        d (int): Constante

    Returns:
        tuple: Valeur min/max associé aux constantes
    """
    xmin = math.inf
    ymin = math.inf
    xmax = - math.inf
    ymax = - math.inf

    cases_touchees = 0
    grille_test = np.zeros((50, 50))

    for n in range (0, 50000):
        if n == 0:
            xn = 0.1
            yn = 0.1

        if mode == 1:
            xn_1 = math.sin(a*yn) + c*math.cos(a*xn)
            yn_1 = math.sin(b*xn) + d*math.cos(b*yn)
        else:
            xn_1 = math.sin(a*yn) - math.cos(b*xn)
            yn_1 = math.sin(c*xn) - math.cos(d*yn)

        xn = xn_1
        yn = yn_1

        if abs(xn) > 10 or abs(yn) > 10:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        if n > 1000:
            if xn > xmax:
                xmax  = xn
            elif xn < xmin:
                xmin = xn
            elif yn > ymax:
                ymax = yn
            elif yn < ymin:
                ymin = yn

        gx = int((xn + 3.0) / 6.0 * 49)
        gy = int((yn + 3.0) / 6.0 * 49)
        if 0 <= gx < 50 and 0 <= gy < 50:
            grille_test[gy, gx] = 1

    for l in range(50):
        for c_idx in range(50):
            if grille_test[l, c_idx] == 1:
                cases_touchees += 1

    return xmin, xmax, ymin, ymax, cases_touchees


def generation_cons(mode: int) -> tuple:
    """Fonction qui génère des constantes menant à une situation non convergantes

    Args:
        mode (int): Type d'équation

    Returns:
        tuple: Valeurs des constantes et des min/max associés
    """
    while True:
        a, b, c, d = init_cons(mode)
        xmin, xmax, ymin, ymax, cases = init(mode, a, b, c, d)

        if (xmax - xmin >= 1) and (ymax - ymin >= 1) and (cases >= 250):
            return a, b, c, d, xmin, xmax, ymin, ymax
        

@jit(nopython=True, fastmath=True)
def calcul(a: int, b: int, c: int, d: int, xmin: int, xmax: int, ymin: int, ymax: int, mode: int):
    """Fonction de calcul de la matrice

    Args:
        a (int): Constante
        b (int): Constante
        c (int): Constante
        d (int): Constante
        xmin (int): Constante
        xmax (int): Constante
        ymin (int): Constante
        ymax (int): Constante
        mode (int): Type d'équation

    Returns:
        Matrice: Résultat des calculs
    """
    matrice = np.zeros((HEIGHT, WIDTH))

    for n in range(0, N-1):
        if n == 0:
            xn = 0.1
            yn = 0.1

        if mode ==  1:
            xn_1 = math.sin(a*yn) + c*math.cos(a*xn)
            yn_1 = math.sin(b*xn) + d*math.cos(b*yn)
        else:   
            xn_1 = math.sin(a*yn) - math.cos(b*xn)
            yn_1 = math.sin(c*xn) - math.cos(d*yn)

        xn = xn_1
        yn = yn_1
        
        i = int((xn - xmin) / (xmax - xmin) * (WIDTH - 1))
        j = int((yn - ymin) / (ymax - ymin) * (HEIGHT - 1))

        if 0 <= i <= WIDTH and 0 <= j <= HEIGHT:
            matrice[j, i] += 1

    return matrice


def color(normalise):
    """Fonction qui choisi aléatoirement la couleur appliquée à l'image

    Args:
        normalise (matrice): Matrice normalisé

    Returns:
        tuple: Matrice, coloration choisie
    """
    n = np.random.randint(0, 2)
    if n:
        palette =  np.random.choice(palettes)
        cmap = plt.get_cmap(palette)

        return (cmap(normalise)[:, :, :3] * 255).astype(np.uint8), palette
    else:
        h = np.random.random()
        r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        couleur_rgb = np.array([int(r * 255), int(g * 255), int(b * 255)], dtype=np.uint8)

        return (normalise[:, :, np.newaxis] * couleur_rgb).astype(np.uint8), f"{couleur_rgb[0]}{couleur_rgb[1]}{couleur_rgb[2]}"


def main():
    """Fonction principale qui génère une image"""
    lst = [1, 1, 1, 0]
    mode = np.random.choice(lst)

    a, b, c, d, xmin, xmax, ymin, ymax = generation_cons(mode)

    matrice = calcul(a, b, c, d, xmin, xmax, ymin, ymax, mode)
    matrice_log = np.log1p(matrice)
    normalise = matrice_log/matrice_log.max()

    image_rgb, style = color(normalise)

    with Image.fromarray(image_rgb) as im:
        im.save(f"Image_{mode}_{style}_{abs(a*b*c*d)}.png")
    print(f"Image_{mode}_{style}_{abs(a*b*c*d)}.png")
    print("Image Gérérée avec succés.")


# Fonction de test
if __name__ == '__main__':
    main()



        
        



