# Importation des modules nécessaires
from numba import jit, njit, prange
import numpy as np
import matplotlib.pyplot as plt
import math
import colorsys
import cv2
from collections import deque
from tqdm import tqdm
import subprocess


# Constantes
WIDTH = 1080
HEIGHT = 1920
N = 10000000

# Couleur possible avec le cmap
palettes = ['CMRmap', 'afmhot', 'copper', 'cubehelix', 'gist_earth' ,'gist_grey', 'gist_heat', 'gist_stern', 'gnuplot', 'gnuplot2', 'hot', 'inferno', 'magma', 'nipy_spectral']


def init_cons(mode: int) -> tuple:
    """Fonction de génération pseudo aléatoire des constantes a, b, c, d

    Args:
        mode (int): Deux intervalles différents suivant les équations choisies

    Returns:
        tuple (int): Les quatre constantes
    """
    if mode == 1:
        a, b, c, d = np.random.uniform(-3.0, 3.0, size=4)
    else:
        a, b, c, d = np.random.uniform(-2*np.pi, 2*np.pi, size=4)

    return a, b, c, d


@jit(nopython=True, fastmath=True, cache=True)
def init(mode: int, a: int, b: int, c: int, d: int) -> tuple:
    """Fonction qui calcul les valeurs min/max et le nombre de cases touchées sur une grille de test de 50x50

    Args:
        mode (int): Deux types d'équations différentes
        a (int): constante
        b (int): constante
        c (int): constante
        d (int): constante

    Returns:
        tuple: Les valeurs min/max associées aux constantes et le nombre de cases touchées
    """
    # Initialisation des valeurs
    xmin = math.inf
    ymin = math.inf
    xmax = - math.inf
    ymax = - math.inf

    cases_touchees = 0
    # Grille de "test" pour empêcher les problèmes de convergence
    grille_test = np.zeros((50, 50))

    xn = 0.1
    yn = 0.1

    # Boucle de test sur les 50000 premiers points
    for n in range (0, 50000):
        # Deux types d'équations possibles
        if mode == 1:
            xn_1 = math.sin(a*yn) + c*math.cos(a*xn)
            yn_1 = math.sin(b*xn) + d*math.cos(b*yn)
        else:
            xn_1 = math.sin(a*yn) - math.cos(b*xn)
            yn_1 = math.sin(c*xn) - math.cos(d*yn)

        xn = xn_1
        yn = yn_1

        # Met a jour les min/max
        if n > 1000:
            if xn > xmax:
                xmax  = xn
            if xn < xmin:
                xmin = xn
            if yn > ymax:
                ymax = yn
            if yn < ymin:
                ymin = yn

            # Remplissage de la grille de test
            gx = int((xn + 3.0) / 6.0 * 49)
            gy = int((yn + 3.0) / 6.0 * 49)
            if 0 <= gx < 50 and 0 <= gy < 50:
                grille_test[gy, gx] = 1

    # Compte le nombre de cases touchées dans la grille
    for l in range(50):
        for c_idx in range(50):
            if grille_test[l, c_idx] == 1:
                cases_touchees += 1

    return xmin, xmax, ymin, ymax, cases_touchees
        

def generation_abcd(mode: int) -> tuple:
    """Génération des quatre constantes qui donne une situation divergente obligatoirement

    Args:
        mode (int): Deux types d'équations

    Returns:
        tuple: Les quatre constantes trouvées
    """
    while True:
        a, b, c, d = init_cons(mode)

        *_, cases = init(mode, a, b, c, d)

        if (cases >= 250):
            return a, b, c, d


def generation_val_valide(mode: int, nb_frame: int) -> tuple:
    """Création des variations des quatre valeurs (a,b,c,d) et on retourne les cas ou ça diverge

    Args:
        mode (int): Deux types d'équations différentes
        nb_frame (int): Nombre de frames de la vidéo

    Returns:
        tuple: Quatre listes des valeurs de a,b,c,d et une liste des valeurs associées min/max
    """
    a_v, b_v, c_v, d_v, e_v = [], [], [], [], []
    # On accepte 20 images de pertes au total
    while (nb_frame - len(a_v) > 20) and (nb_frame - len(b_v) > 20) and (nb_frame - len(c_v) > 20) and (nb_frame - len(d_v) > 20):
        a_v, b_v, c_v, d_v = [], [], [], []
        # Génération des constantes a,b,c,d
        a, b, c, d = generation_abcd(mode)

        # Génération des variations de a,b,c,d
        a_values = np.linspace(a-0.3, a+0.3, nb_frame)
        b_values = np.linspace(b-0.3, b+0.3, nb_frame)
        c_values = np.linspace(c-0.3, c+0.3, nb_frame)
        d_values = np.linspace(d-0.3, d+0.3, nb_frame)

        # On ajoute seulement les valeurs qui ne font pas converger 
        for (a, b, c, d) in zip(a_values, b_values, c_values, d_values):
            xmin, xmax, ymin, ymax, cases = init(mode, a, b, c, d)
            if (cases >= 250):
                a_v.append((a))
                b_v.append((b))
                c_v.append((c))
                d_v.append((d))
                e_v.append((xmin, xmax, ymin, ymax))

    return a_v, b_v, c_v, d_v, e_v


@njit(parallel=True, fastmath=True, cache=True)
def calcul(a: int, b: int, c: int, d: int, xmin: int, xmax: int, ymin: int, ymax: int, mode: int):
    """Fonction principale qui calcul la matrice finale

    Args:
        a (int): constante
        b (int): constante
        c (int): constante
        d (int): constante
        xmin (int): constante
        xmax (int): constante
        ymin (int): constante
        ymax (int): constante
        mode (int): Deux types d'équations différentes

    Returns:
        Np.array: Matrice de sortie
    """
    n_traj = 12                    
    par = N // n_traj
    # On intialise une matrice ou chaque thread écrira dedans
    locales = np.zeros((n_traj, HEIGHT, WIDTH), dtype=np.int32)
    sx = (WIDTH - 1) / (xmax - xmin)
    sy = (HEIGHT - 1) / (ymax - ymin)

    # On lance le calcul en parallèle sur les 6 coeurs
    for t in prange(n_traj):
        # On prend des directions légèrement diférrentes
        xn = 0.1 + 0.013 * t
        yn = 0.1 - 0.011 * t
        for k in range(par + 200):
            # Les deux équations différentes
            if mode == 1:
                x1 = math.sin(a*yn) + c*math.cos(a*xn)
                y1 = math.sin(b*xn) + d*math.cos(b*yn)
            else:
                x1 = math.sin(a*yn) - math.cos(b*xn)
                y1 = math.sin(c*xn) - math.cos(d*yn)
            xn, yn = x1, y1
            # On ne prend pas en compte la phase de transition
            if k >= 200:
                i = int((xn - xmin) * sx)
                j = int((yn - ymin) * sy)
                if 0 <= i < WIDTH and 0 <= j < HEIGHT:
                    locales[t, j, i] += 1

    # On complète la matrice finale  
    out = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
    for j in prange(HEIGHT):
        for i in range(WIDTH):
            s = 0
            for t in range(n_traj):
                s += locales[t, j, i]
            out[j, i] = s
    return out


def init_color() -> tuple:
    """Choix pseudo aléatoire de la coloration

    Returns:
        tuple: Un entier donne quelle coloration est choisi et une autre le tableau/string
    """
    n = np.random.randint(0, 2)
    if n:
        palette =  np.random.choice(palettes)
        return 0, palette
        
    else:
        h = np.random.random()
        r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        couleur_rgb = np.array([int(r * 255), int(g * 255), int(b * 255)], dtype=np.uint8)
        return 1, couleur_rgb


def construire_lut(theme: int, couleur):
    """Fonction qui créer la lut, permet de convertir la matrice en couleur

    Args:
        theme (int): Deux thèmes de couleur différent
        couleur (Array/Str): Dépend du thème choisi, généré dans 'init_color'

    Returns:
        Array: Tableau qui sera utilisé par la suite pour le rendu
    """

    t = np.linspace(0, 1, 256)
    if theme:
        lut = t[:, None] * couleur[None, :]
    else:
        lut = plt.get_cmap(couleur)(t)[:, :3] * 255
    return lut.astype(np.float32)


@njit(parallel=True, cache=True)
def somme_ponderee(m0, m1, m2, m3, m4, out):
    """Fonction qui fait une moyenne de 5 images avec une pondération différente selon l'ordre des images, rendu plus net

    Args:
        m0 (array): image 1
        m1 (array): image 2
        m2 (array): image 3
        m3 (array): image 4
        m4 (array): image 5
        out (array): image finale
    """
    H, W = out.shape
    for j in prange(H):
        for i in range(W):
            out[j, i] = m0[j, i] + 2*m1[j, i] + 3*m2[j, i] + 2*m3[j, i] + m4[j, i]


@njit(parallel=True, cache=True)
def appliquer_table(somme, table, p: int, img):
    """Fonction pour appliquer table a la somme

    Args:
        somme (array): matrice calculée 
        table (array): matrice des valeurs possibles
        p (int): Valeur de saturation
        img (array): on copie la couleur dans cet matrice
    """
    H, W = somme.shape
    for j in prange(H):
        for i in range(W):
            v = somme[j, i]
            if v > p:
                v = p
            img[j, i, 0] = table[v, 0]
            img[j, i, 1] = table[v, 1]
            img[j, i, 2] = table[v, 2]


def rendu(somme, lut_bgr, net, force_glow=0.7):
    """Rendu de l'image a partir de la matrice calculée

    Args:
        somme (array): matrice finale après pondération
        lut_bgr (array/): la lut pour créer la couleur
        net (array): matrice qui sert au glow
        force_glow (float, optional): Force de l'effet de glowing. Defaults to 0.7.

    Returns:
        img: Image avec rendu finale
    """
    # On prend pas tous les points de l'image pour reduire le temps de rendu
    s = somme[::3, ::3]
    nz = s[s > 0]

    if nz.size == 0:
        return None

    # Valeur de saturation
    p = max(1, int(np.ceil(np.percentile(nz, 99.5))))

    # On liste toutes les valeurs possibles de chaque pixels  
    idx = np.arange(p + 1)
    n = np.clip(np.log1p(idx / 9.0) / np.log1p(p / 9.0), 0, 1) ** 0.8
    # Une couleur BGR par valeur possible de n
    table = np.ascontiguousarray(lut_bgr[(n * 255).astype(np.uint8)])

    # Application du lut/normalisation en parallèle
    appliquer_table(somme, table, p, net)

    # Application de l'effet de glowing
    couverture = np.count_nonzero(somme) / somme.size
    force = force_glow * min(1.0, 0.25 / max(couverture, 1e-6))

    petit = cv2.resize(net, (WIDTH // 4, HEIGHT // 4), interpolation=cv2.INTER_AREA)
    glow = cv2.GaussianBlur(petit, (0, 0), 3)
    glow = cv2.resize(glow, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(net, 1.0, glow, force, 0)


def generer_video(fps, nb_frame):
    """Fonction principale qui génère une vidéo"""
    # Initalisation des constantes
    mode = np.random.randint(0, 2)
    theme, couleur = init_color()

    # Formule de commande pour créer la vidéo
    ff = subprocess.Popen(
    ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
     '-s', f'{WIDTH}x{HEIGHT}', '-r', f'{fps}', '-i', '-',
     '-c:v', 'libx264', '-preset', 'medium', '-crf', '16',
     '-pix_fmt', 'yuv420p', 'video.mp4'],
    stdin=subprocess.PIPE, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    a_v, b_v, c_v, d_v, e_v = generation_val_valide(mode, nb_frame)

    # On définit le cadrage
    bornes = np.array([v[:] for v in e_v]) 
    xmin, ymin = bornes[:, 0].min(), bornes[:, 2].min()
    xmax, ymax = bornes[:, 1].max(), bornes[:, 3].max()

    # On donne une marge de 10%
    mx, my = 0.1 * (xmax - xmin), 0.1 * (ymax - ymin)
    xmin -= mx; xmax += mx; ymin -= my; ymax += my

    # Initialisation des tableaux nécessaires au rendu
    lut_bgr = np.ascontiguousarray(construire_lut(theme, couleur)[:, ::-1]).astype(np.uint8)
    somme = np.empty((HEIGHT, WIDTH), dtype=np.int32)
    net = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)

    # On utiliser une deque pour générer la moyenne pondéré des images
    buf = deque(maxlen=5)

    # Boucle principale sur toutes les valeurs valides de a,b,c,d
    for (a, b, c, d) in tqdm(zip(a_v, b_v, c_v, d_v), desc="🎬 Rendu vidéo", unit=" frames"):
        buf.append(calcul(a, b, c, d, xmin, xmax, ymin, ymax, mode))

        if len(buf) == 5:
            # Moyenne pondéré
            somme_ponderee(buf[0], buf[1], buf[2], buf[3], buf[4], somme)
            # Génération de l'image
            img = rendu(somme, lut_bgr, net)

            # Gestion des images trop sombres/noires
            if img is None or img[::4, ::4].mean() < 15:
                continue
            ff.stdin.write(img.tobytes()) #type: ignore

    ff.stdin.close() #type: ignore
    ff.wait()
    print("Vidéo générée avec succès.")


# Permet les tests
if __name__ == '__main__':
    generer_video(24, 240)


        
        



