# Importation des modules
from mido import MidiFile, Message, MidiTrack, MetaMessage, bpm2tempo
import subprocess
import numpy as np

# Listes des différentes gammes possibles
GAMMES = {
    'pentatonique_mineure': [0, 3, 5, 7, 10],
    'hirajoshi':            [0, 2, 3, 7, 8],
    'dorien':               [0, 2, 3, 5, 7, 9, 10],
    'eolien':               [0, 2, 3, 5, 7, 8, 10],
    'phrygien':             [0, 1, 3, 5, 7, 8, 10],
    'lydien':               [0, 2, 4, 6, 7, 9, 11],
}

# Progressions de basse
PROGRESSIONS = [[0, 3, 4, 2], [0, 5, 3, 4], [0, 2, 3, 4],
                [0, 0, 3, 2], [0, 6, 5, 4], [0, 4, 3, 5]]

# Instruments General MIDI
LEADS = [98, 8, 9, 10, 11, 14, 88] 
DRONES = [89, 91, 92, 94, 95]         
OSTINATOS = [12, 11, 46, 10, 45]


def note(root, gamme, degre):
    """Degré de gamme -> note MIDI (les degrés peuvent dépasser l'octave ou être négatifs)."""
    octave, i = divmod(degre, len(gamme))
    return int(root + 12 * octave + gamme[i])


def marche(rng, degre, bas, haut):
    """Marche aléatoire : on bouge surtout de 1 degré, rarement de 2."""
    pas = rng.choice([-2, -1, 0, 1, 2], p=[0.15, 0.30, 0.10, 0.30, 0.15])
    return int(np.clip(degre + pas, bas, haut))


def nouveau_motif(rng, longueur, haut):
    """Génère un motif aléatoire"""
    d = int(rng.integers(0, haut + 1))
    motif = []
    for _ in range(longueur):
        d = marche(rng, d, 0, haut)
        motif.append(d)
    return motif


def main(seed=None, duree=30, sortie='new_song'):
    """Fonction principale qui génère une musique

    Args:
        seed (int, optional): Seed de l'aléatoire. Defaults to None.
        duree (int, optional): Durée du son en secondes. Defaults to 30.
        sortie (str, optional): Nom du fichier en sortie. Defaults to 'new_song'.
    """
    rng = np.random.default_rng(seed)

    # Paramètres globaux tirés au hasard
    bpm = int(rng.integers(90, 141))
    r = int(rng.integers(0, 12))      
    nom_gamme = rng.choice(list(GAMMES))
    gamme = GAMMES[nom_gamme]
    progression = PROGRESSIONS[rng.integers(len(PROGRESSIONS))] + PROGRESSIONS[rng.integers(len(PROGRESSIONS))]
    prog_lead, prog_drone, prog_osti = (int(rng.choice(L)) for L in (LEADS, DRONES, OSTINATOS))
    print(f"{bpm} bpm | tonique {r} | {nom_gamme} | instruments {prog_lead}/{prog_drone}/{prog_osti}")

    root_basse = 36 + r
    root_mel = 60 + r
    n = len(gamme)

    # On définit le nombre de mesure
    TICKS_PER_MEASURE, croche = 1920, 240
    nb_mesures = max(4, round(duree * bpm / 60 / 4))
    total_ticks = nb_mesures * TICKS_PER_MEASURE

    # On créer des cycles pour les répétitions
    cycles = rng.uniform(0.5, 2.5)
    phase = rng.uniform(0, 2 * np.pi)
    intensite = np.clip(0.55 + 0.45 * np.sin(2 * np.pi * cycles * np.arange(nb_mesures) / nb_mesures + phase), 0.25, 1.0)

    # On créer les tracks midi
    mid = MidiFile()
    t_lead, t_drone, t_osti = MidiTrack(), MidiTrack(), MidiTrack()
    mid.tracks.extend([t_lead, t_drone, t_osti])

    # On initialise les voix / channels
    t_lead.append(MetaMessage('set_tempo', tempo=bpm2tempo(bpm), time=0))
    t_lead.append(Message('program_change', program=prog_lead, channel=0, time=0))
    t_drone.append(Message('program_change', program=prog_drone, channel=1, time=0))
    t_osti.append(Message('program_change', program=prog_osti, channel=2, time=0))

    # Drone : progression de 8 mesures
    for m in range(nb_mesures):
        deg = progression[m % len(progression)]
        notes = [note(root_basse, gamme, deg)]
        if rng.random() < 0.5:
            notes.append(notes[0] + 7)
        for x in notes:
            t_drone.append(Message('note_on', note=x, velocity=int(rng.integers(55, 66)), channel=1, time=0))
        t_drone.append(Message('note_off', note=notes[0], velocity=0, channel=1, time=TICKS_PER_MEASURE))
        for x in notes[1:]:
            t_drone.append(Message('note_off', note=x, velocity=0, channel=1, time=0))

    # Ostinato : un petit motif répété et muté, renouvelé toutes les 8 mesures
    haut_osti = n
    attente, pos, motif = 0, 0, None
    for m in range(nb_mesures):
        if m % 8 == 0:
            motif = nouveau_motif(rng, int(rng.choice([3, 4, 5, 6, 8])), haut_osti)
        for _ in range(8):
            idx = pos % len(motif) #type: ignore
            pos += 1
            if rng.random() < 0.1: # mutation d'une note du motif
                motif[idx] = marche(rng, motif[idx], 0, haut_osti) #type: ignore
            if rng.random() < intensite[m]:
                accent = 8 if idx == 0 else 0 # le début du motif est accentué
                vel = int(rng.integers(38, 52)) + accent
                t_osti.append(Message('note_on', note=note(root_mel, gamme, motif[idx]), #type: ignore
                                      velocity=vel, channel=2, time=attente))
                t_osti.append(Message('note_off', note=note(root_mel, gamme, motif[idx]), #type: ignore
                                      velocity=0, channel=2, time=croche))
                attente = 0
            else:
                attente += croche

    # Lead : marche aléatoire + silences + durées pondérées
    haut_lead = n + 3
    deg = int(rng.integers(0, haut_lead + 1))
    ticks, attente = 0, 0
    while ticks < total_ticks:
        d = int(rng.choice([240, 480, 960, 1440], p=[0.15, 0.35, 0.30, 0.20]))
        d = min(d, total_ticks - ticks)
        m = min(ticks // TICKS_PER_MEASURE, nb_mesures - 1)
        if rng.random() < 0.35 * (1.2 - intensite[m]):   # plus de silences quand l'intensité baisse
            attente += d
        else:
            deg = marche(rng, deg, 0, haut_lead)
            x = note(root_mel, gamme, deg)
            t_lead.append(Message('note_on', note=x, velocity=int(rng.integers(55, 75)), channel=0, time=attente))
            t_lead.append(Message('note_off', note=x, velocity=0, channel=0, time=d))
            attente = 0
        ticks += d

    mid.save(f'{sortie}.mid')

    cmd = ["fluidsynth", "-ni", "-F", "audio.wav", "sys/GeneralUser-GS.sf2", f'{sortie}.mid']
    resultat = subprocess.run(cmd, capture_output=True, text=True)
    if not resultat.returncode:
        print("Audio généré avec succès")

# Fonction de test
if __name__ == '__main__':
    main()